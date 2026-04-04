# Agent Factory API Reference (INFRA-010)

## Overview

The Agent Factory provides a complete platform for agent registration, execution, lifecycle management, package distribution (Agent Hub), execution queue management, and async operations. It covers the full agent lifecycle from creation through deployment, monitoring, and retirement.

**Router Prefixes:**
- `/api/v1/factory` -- Core agent CRUD and execution
- `/api/v1/factory/lifecycle` -- Lifecycle management (deploy, rollback, drain, retire)
- `/api/v1/factory/operations` -- Async 202 Accepted operations
- `/api/v1/factory/hub` -- Agent Hub package registry
- `/api/v1/factory/queue` -- Execution queue management

**Tags:** `Agent Factory`, `Agent Lifecycle`, `Operations`, `Agent Hub`, `Execution Queue`
**Source:** `greenlang/infrastructure/agent_factory/api/`

---

## Endpoint Summary

### Core Agent CRUD

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| POST | `/api/v1/factory/agents` | Register a new agent | Yes |
| GET | `/api/v1/factory/agents` | List agents (paginated) | Yes |
| GET | `/api/v1/factory/agents/{key}` | Get agent details | Yes |
| PUT | `/api/v1/factory/agents/{key}` | Update agent configuration | Yes |
| DELETE | `/api/v1/factory/agents/{key}` | Deregister an agent | Yes |
| POST | `/api/v1/factory/agents/{key}/execute` | Trigger agent execution | Yes |
| GET | `/api/v1/factory/agents/{key}/metrics` | Get agent metrics | Yes |
| POST | `/api/v1/factory/agents/batch-execute` | Execute multiple agents | Yes |

### Lifecycle Management

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| POST | `/api/v1/factory/lifecycle/agents/{key}/deploy` | Deploy agent (async 202) | Yes |
| POST | `/api/v1/factory/lifecycle/agents/{key}/rollback` | Rollback agent (async 202) | Yes |
| POST | `/api/v1/factory/lifecycle/agents/{key}/drain` | Drain agent | Yes |
| POST | `/api/v1/factory/lifecycle/agents/{key}/retire` | Retire agent permanently | Yes |
| GET | `/api/v1/factory/lifecycle/agents/{key}/health` | Get agent health | Yes |
| POST | `/api/v1/factory/lifecycle/agents/{key}/restart` | Restart agent | Yes |
| GET | `/api/v1/factory/lifecycle/agents/{key}/history` | Get lifecycle history | Yes |

### Async Operations

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| POST | `/api/v1/factory/operations/` | Create async operation (202) | Yes |
| GET | `/api/v1/factory/operations/{operation_id}` | Poll operation status | Yes |
| DELETE | `/api/v1/factory/operations/{operation_id}` | Cancel operation | Yes |
| GET | `/api/v1/factory/operations/` | List operations | Yes |

### Agent Hub

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/api/v1/factory/hub/packages` | Search packages | Yes |
| GET | `/api/v1/factory/hub/packages/{key}` | Get package details | Yes |
| GET | `/api/v1/factory/hub/packages/{key}/versions` | List package versions | Yes |
| POST | `/api/v1/factory/hub/packages` | Publish a package (multipart) | Yes |
| DELETE | `/api/v1/factory/hub/packages/{key}/{version}` | Unpublish a version | Yes |
| GET | `/api/v1/factory/hub/packages/{key}/{version}/download` | Download a package | Yes |

### Execution Queue

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/api/v1/factory/queue/status` | Queue status | Yes |
| GET | `/api/v1/factory/queue/tasks` | List tasks | Yes |
| GET | `/api/v1/factory/queue/tasks/{task_id}` | Get task details | Yes |
| POST | `/api/v1/factory/queue/tasks/{task_id}/retry` | Retry a failed task | Yes |
| POST | `/api/v1/factory/queue/tasks/{task_id}/cancel` | Cancel a queued task | Yes |
| GET | `/api/v1/factory/queue/dlq` | List dead-letter queue | Yes |
| POST | `/api/v1/factory/queue/dlq/{dlq_id}/reprocess` | Reprocess DLQ item | Yes |

---

## Core Agent Endpoints

### POST /api/v1/factory/agents

Register a new agent in the factory.

**Request Body:**

```json
{
  "agent_key": "carbon-calc-v2",
  "version": "2.1.0",
  "agent_type": "deterministic",
  "description": "Scope 1-3 carbon emissions calculator agent.",
  "entry_point": "agent.py",
  "config": {
    "timeout_seconds": 60,
    "max_retries": 3
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `agent_key` | string | Yes | Unique agent key (3-64 chars) |
| `version` | string | No | Semantic version (default: `"0.1.0"`) |
| `agent_type` | string | No | Agent type: `deterministic`, `reasoning`, `insight` |
| `description` | string | No | Agent description |
| `entry_point` | string | No | Entry-point module (default: `"agent.py"`) |
| `config` | object | No | Agent configuration |

**Response (201 Created):**

```json
{
  "agent_key": "carbon-calc-v2",
  "version": "2.1.0",
  "agent_type": "deterministic",
  "description": "Scope 1-3 carbon emissions calculator agent.",
  "status": "created",
  "created_at": "2026-04-04T11:00:00Z",
  "updated_at": "2026-04-04T11:00:00Z",
  "config": {"timeout_seconds": 60, "max_retries": 3}
}
```

**Error Responses:**

| Status | Description |
|--------|-------------|
| 409 | Agent with this key already registered |

---

### POST /api/v1/factory/agents/{key}/execute

Trigger an agent execution. Returns immediately with a task ID for status polling.

**Request Body:**

```json
{
  "input_data": {
    "facility_id": "FAC-001",
    "reporting_year": 2025,
    "scope": "scope1"
  },
  "priority": 0,
  "timeout_seconds": 60,
  "correlation_id": "req-abc123"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input_data` | object | Yes | Input payload for the agent |
| `priority` | integer | No | Priority 0 (highest) to 10 (lowest) |
| `timeout_seconds` | integer | No | Execution timeout (1-300, default: 30) |
| `correlation_id` | string | No | Optional correlation ID for tracing |

**Response (200 OK):**

```json
{
  "task_id": "task_a1b2c3d4e5f6",
  "agent_key": "carbon-calc-v2",
  "status": "queued",
  "result": null,
  "provenance_hash": null,
  "duration_ms": null,
  "correlation_id": "req-abc123"
}
```

---

### GET /api/v1/factory/agents/{key}/metrics

Get execution metrics for an agent.

**Response (200 OK):**

```json
{
  "agent_key": "carbon-calc-v2",
  "execution_count": 15420,
  "error_count": 23,
  "success_rate": 99.85,
  "avg_duration_ms": 342.5,
  "p95_duration_ms": 890.0,
  "p99_duration_ms": 1250.0,
  "total_cost_usd": 45.67,
  "queue_depth": 3,
  "active_instances": 4
}
```

---

## Lifecycle Endpoints

### POST /api/v1/factory/lifecycle/agents/{key}/deploy

Deploy an agent to the target environment. Uses the async 202 pattern -- returns immediately with a poll URL.

**Request Body:**

```json
{
  "version": "2.1.0",
  "environment": "staging",
  "strategy": "canary",
  "canary_percent": 10,
  "config_overrides": {"log_level": "DEBUG"}
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | Yes | Version to deploy |
| `environment` | string | No | Target: `dev`, `staging`, `prod` (default: `dev`) |
| `strategy` | string | No | Strategy: `rolling`, `canary`, `blue-green` (default: `rolling`) |
| `canary_percent` | integer | No | Canary traffic percentage (1-50, default: 5) |
| `config_overrides` | object | No | Per-deployment configuration overrides |

**Response (202 Accepted):**

```json
{
  "operation_id": "op_abc123def456",
  "operation_type": "deploy",
  "agent_key": "carbon-calc-v2",
  "status": "pending",
  "progress_pct": 0,
  "poll_url": "https://api.greenlang.io/api/v1/factory/operations/op_abc123def456",
  "started_at": null,
  "completed_at": null,
  "result": null,
  "error_message": null
}
```

---

### GET /api/v1/factory/lifecycle/agents/{key}/health

Get the health status of an agent, including liveness, readiness, and dependency checks.

**Response (200 OK):**

```json
{
  "agent_key": "carbon-calc-v2",
  "status": "healthy",
  "liveness": "pass",
  "readiness": "pass",
  "checks": {
    "database": {"status": "pass", "latency_ms": 2.1},
    "redis": {"status": "pass", "latency_ms": 0.8}
  },
  "last_check_at": "2026-04-04T11:20:00Z"
}
```

---

## Async Operations

All long-running operations (deploy, rollback, pack, publish, migrate) use the 202 Accepted pattern with idempotency support.

### POST /api/v1/factory/operations/

Create a new async operation.

**Request Body:**

```json
{
  "operation_type": "deploy",
  "agent_key": "carbon-calc-v2",
  "params": {"version": "2.1.0", "environment": "staging"},
  "idempotency_key": "deploy-carbon-calc-v2-staging-2026-04-04"
}
```

**Valid operation types:** `deploy`, `rollback`, `pack`, `publish`, `migrate`

**Response (202 Accepted):** Returns `OperationResponse` with `poll_url` for status tracking.

### GET /api/v1/factory/operations/{operation_id}

Poll the status and progress of an operation. Progress is reported as `progress_pct` (0-100).

### DELETE /api/v1/factory/operations/{operation_id}

Request cooperative cancellation of a running or pending operation. The handler checks a cancellation flag at each phase boundary.

---

## Agent Hub

### GET /api/v1/factory/hub/packages

Search the Agent Hub for packages.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | - | Search query string |
| `tags` | string | - | Comma-separated tags to filter by |
| `agent_type` | string | - | Filter by agent type |
| `page` | integer | 1 | Page number |
| `page_size` | integer | 20 | Items per page |

**Response (200 OK):**

```json
{
  "packages": [
    {
      "package_key": "carbon-calc",
      "latest_version": "2.1.0",
      "description": "Scope 1-3 carbon emissions calculator agent.",
      "agent_type": "deterministic",
      "tags": ["carbon", "emissions", "scope1", "scope2", "scope3"],
      "total_downloads": 5420,
      "published_at": "2026-04-01T00:00:00Z"
    }
  ],
  "total": 3,
  "page": 1,
  "page_size": 20
}
```

### POST /api/v1/factory/hub/packages

Publish a new package or version to the Agent Hub. Uses multipart form upload.

**Form Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `package` | file | Yes | The `.glpack` archive |
| `version` | string | Yes | Semantic version |
| `checksum` | string | No | SHA-256 checksum for validation |
| `tag` | string | No | Version tag (e.g., `latest`, `beta`) |

---

## Execution Queue

### GET /api/v1/factory/queue/status

Get aggregate queue status including depth, worker count, throughput, and DLQ depth.

**Response (200 OK):**

```json
{
  "total_depth": 12,
  "active_workers": 4,
  "idle_workers": 0,
  "throughput_per_min": 42.5,
  "avg_wait_ms": 155.2,
  "oldest_task_age_s": 12.3,
  "dlq_depth": 1
}
```

### POST /api/v1/factory/queue/tasks/{task_id}/retry

Retry a failed task. Creates a new task with the same input. Only tasks with status `failed` can be retried.

### POST /api/v1/factory/queue/dlq/{dlq_id}/reprocess

Reprocess a dead-letter queue item by creating a new task and removing the item from the DLQ.
