# GL-001 ThermalCommand Orchestrator API

The ThermalCommand Orchestrator (GL-001) is the central coordination agent for all process heat operations. This API provides endpoints for submitting calculation workflows, monitoring job status, and retrieving results.

---

## Table of Contents

1. [Submit Calculation Workflow](#submit-calculation-workflow)
2. [Get Job Status](#get-job-status)
3. [Get Job Results](#get-job-results)
4. [Cancel Job](#cancel-job)
5. [List Workflows](#list-workflows)
6. [Agent Management](#agent-management)
7. [Error Codes](#error-codes)

---

## Submit Calculation Workflow

Submit a new calculation workflow for execution.

```http
POST /api/v1/process-heat/calculate
```

### Request Headers

| Header | Required | Description |
|--------|----------|-------------|
| `Authorization` | Yes | Bearer token |
| `Content-Type` | Yes | `application/json` |
| `X-Request-ID` | Recommended | Unique request identifier |
| `X-Idempotency-Key` | Recommended | Idempotency key for retry safety |

### Request Body

```json
{
  "workflow_type": "optimization",
  "name": "Boiler Efficiency Optimization",
  "priority": "high",
  "timeout_s": 300,
  "parameters": {
    "boiler_id": "BLR-001",
    "target_efficiency_pct": 92.5,
    "optimization_mode": "efficiency",
    "constraints": {
      "max_stack_temp_f": 400,
      "min_o2_pct": 2.0,
      "max_o2_pct": 4.0
    }
  },
  "tasks": [
    {
      "task_type": "efficiency_calculation",
      "name": "Calculate Current Efficiency",
      "target_agent_type": "GL-002",
      "inputs": {
        "fuel_type": "natural_gas",
        "fuel_flow_rate": 150.0,
        "steam_flow_rate_lb_hr": 25000,
        "steam_pressure_psig": 150,
        "feedwater_temperature_f": 220,
        "flue_gas_o2_pct": 3.2,
        "flue_gas_temperature_f": 380
      }
    },
    {
      "task_type": "combustion_optimization",
      "name": "Optimize Combustion",
      "target_agent_type": "GL-005",
      "inputs": {}
    }
  ],
  "rollback_enabled": true,
  "checkpoint_enabled": true,
  "metadata": {
    "plant_id": "PLANT-001",
    "requested_by": "operator@example.com"
  }
}
```

### Request Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `workflow_type` | string | Yes | Workflow type: `optimization`, `monitoring`, `compliance`, `maintenance`, `emergency`, `calibration`, `reporting` |
| `name` | string | Yes | Human-readable workflow name |
| `priority` | string | No | Priority level: `low`, `normal`, `high`, `critical`, `emergency`. Default: `normal` |
| `timeout_s` | float | No | Timeout in seconds (1-86400). Default: 300 |
| `parameters` | object | No | Workflow-specific parameters |
| `tasks` | array | No | Optional explicit task specifications |
| `rollback_enabled` | boolean | No | Enable rollback on failure. Default: true |
| `checkpoint_enabled` | boolean | No | Enable workflow checkpointing. Default: true |
| `required_agents` | array | No | Required agent types for execution |
| `metadata` | object | No | Additional metadata for tracking |

### Task Specification

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task_type` | string | Yes | Task type identifier |
| `name` | string | Yes | Human-readable task name |
| `target_agent_type` | string | Yes | Target agent type (e.g., `GL-002`) |
| `target_agent_id` | string | No | Specific agent ID (optional) |
| `priority` | string | No | Task priority. Default: inherits from workflow |
| `timeout_s` | float | No | Task timeout (1-3600). Default: 60 |
| `inputs` | object | No | Task input parameters |
| `retry_count` | integer | No | Retry count (0-5). Default: 2 |

### Response (202 Accepted)

```json
{
  "success": true,
  "data": {
    "workflow_id": "wf_a1b2c3d4e5f6",
    "status": "pending",
    "name": "Boiler Efficiency Optimization",
    "priority": "high",
    "tasks_total": 2,
    "created_at": "2025-12-06T10:30:00Z",
    "estimated_completion_s": 45
  },
  "metadata": {
    "request_id": "req_xyz789",
    "processing_time_ms": 12
  },
  "timestamp": "2025-12-06T10:30:00Z"
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `workflow_id` | string | Unique workflow identifier |
| `status` | string | Initial status: `pending` |
| `name` | string | Workflow name |
| `priority` | string | Assigned priority |
| `tasks_total` | integer | Total number of tasks |
| `created_at` | string | ISO 8601 creation timestamp |
| `estimated_completion_s` | float | Estimated completion time in seconds |

### Example (cURL)

```bash
curl -X POST "https://api.greenlang.io/v1/process-heat/calculate" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: req-$(uuidgen)" \
  -d '{
    "workflow_type": "optimization",
    "name": "Boiler Efficiency Optimization",
    "priority": "high",
    "parameters": {
      "boiler_id": "BLR-001",
      "target_efficiency_pct": 92.5
    }
  }'
```

### Example (Python)

```python
import requests

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json",
    "X-Request-ID": "req-12345"
}

payload = {
    "workflow_type": "optimization",
    "name": "Boiler Efficiency Optimization",
    "priority": "high",
    "parameters": {
        "boiler_id": "BLR-001",
        "target_efficiency_pct": 92.5,
        "optimization_mode": "efficiency"
    }
}

response = requests.post(
    "https://api.greenlang.io/v1/process-heat/calculate",
    headers=headers,
    json=payload
)

result = response.json()
workflow_id = result["data"]["workflow_id"]
print(f"Workflow submitted: {workflow_id}")
```

---

## Get Job Status

Retrieve the current status of a workflow job.

```http
GET /api/v1/process-heat/status/{job_id}
```

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | Workflow job identifier |

### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `include_tasks` | boolean | No | Include task-level status. Default: false |
| `include_metrics` | boolean | No | Include performance metrics. Default: false |

### Response (200 OK)

```json
{
  "success": true,
  "data": {
    "workflow_id": "wf_a1b2c3d4e5f6",
    "status": "running",
    "name": "Boiler Efficiency Optimization",
    "priority": "high",
    "start_time": "2025-12-06T10:30:05Z",
    "duration_ms": 12500,
    "tasks_completed": 1,
    "tasks_failed": 0,
    "tasks_total": 2,
    "current_task": {
      "task_id": "task_789xyz",
      "name": "Optimize Combustion",
      "status": "running",
      "assigned_agent": "GL-005-001",
      "progress_pct": 45
    },
    "checkpoints": [
      "checkpoint_001"
    ]
  },
  "metadata": {
    "request_id": "req_abc123"
  },
  "timestamp": "2025-12-06T10:30:18Z"
}
```

### Status Values

| Status | Description |
|--------|-------------|
| `pending` | Workflow is queued for execution |
| `running` | Workflow is currently executing |
| `paused` | Workflow is paused (user-initiated) |
| `completed` | Workflow completed successfully |
| `failed` | Workflow failed |
| `cancelled` | Workflow was cancelled |
| `timeout` | Workflow timed out |

### Response with Tasks (include_tasks=true)

```json
{
  "success": true,
  "data": {
    "workflow_id": "wf_a1b2c3d4e5f6",
    "status": "running",
    "tasks": [
      {
        "task_id": "task_123abc",
        "name": "Calculate Current Efficiency",
        "status": "completed",
        "assigned_agent": "GL-002-001",
        "start_time": "2025-12-06T10:30:05Z",
        "end_time": "2025-12-06T10:30:10Z",
        "duration_ms": 5000,
        "retry_count": 0
      },
      {
        "task_id": "task_789xyz",
        "name": "Optimize Combustion",
        "status": "running",
        "assigned_agent": "GL-005-001",
        "start_time": "2025-12-06T10:30:11Z",
        "end_time": null,
        "duration_ms": 7500,
        "retry_count": 0
      }
    ]
  },
  "timestamp": "2025-12-06T10:30:18Z"
}
```

### Example (cURL)

```bash
curl -X GET "https://api.greenlang.io/v1/process-heat/status/wf_a1b2c3d4e5f6?include_tasks=true" \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

---

## Get Job Results

Retrieve the results of a completed workflow job.

```http
GET /api/v1/process-heat/results/{job_id}
```

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | Workflow job identifier |

### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `format` | string | No | Response format: `json` (default), `csv`, `pdf` |
| `include_provenance` | boolean | No | Include provenance chain. Default: false |
| `include_uncertainty` | boolean | No | Include uncertainty analysis. Default: true |

### Response (200 OK)

```json
{
  "success": true,
  "data": {
    "workflow_id": "wf_a1b2c3d4e5f6",
    "status": "completed",
    "name": "Boiler Efficiency Optimization",
    "start_time": "2025-12-06T10:30:05Z",
    "end_time": "2025-12-06T10:30:45Z",
    "duration_ms": 40000,
    "tasks_completed": 2,
    "tasks_failed": 0,
    "tasks_total": 2,
    "final_output": {
      "efficiency": {
        "gross_efficiency_pct": 85.7,
        "net_efficiency_pct": 83.2,
        "combustion_efficiency_pct": 88.5,
        "total_losses_pct": 14.3,
        "loss_breakdown": {
          "dry_flue_gas_loss_pct": 8.2,
          "moisture_from_h2_loss_pct": 3.8,
          "radiation_loss_pct": 1.5,
          "blowdown_loss_pct": 0.8
        },
        "uncertainty": {
          "lower_bound_pct": 84.1,
          "upper_bound_pct": 87.3,
          "confidence_level": 0.95
        }
      },
      "recommendations": [
        {
          "recommendation_id": "rec_001",
          "category": "combustion",
          "priority": "high",
          "title": "Reduce Excess Air",
          "description": "Current O2 at 3.2% indicates excess air of 17%. Reducing to 2.5% O2 would improve efficiency.",
          "current_value": 3.2,
          "recommended_value": 2.5,
          "estimated_savings_pct": 1.2,
          "estimated_annual_savings_usd": 45000,
          "implementation_difficulty": "low",
          "requires_shutdown": false
        },
        {
          "recommendation_id": "rec_002",
          "category": "economizer",
          "priority": "medium",
          "title": "Clean Economizer",
          "description": "Economizer effectiveness degraded to 62% from design 70%. Cleaning recommended.",
          "current_value": 0.62,
          "recommended_value": 0.70,
          "estimated_savings_pct": 0.8,
          "estimated_annual_savings_usd": 28000,
          "implementation_difficulty": "medium",
          "requires_shutdown": true
        }
      ],
      "kpis": {
        "heat_rate_btu_kwh": 10500,
        "steam_to_fuel_ratio": 8.5,
        "stack_temperature_f": 380,
        "excess_air_pct": 17.2,
        "co_emissions_ppm": 45
      }
    },
    "task_results": {
      "task_123abc": {
        "task_id": "task_123abc",
        "status": "completed",
        "output": {
          "gross_efficiency_pct": 85.7,
          "calculation_method": "ASME_PTC_4.1_LOSSES"
        }
      },
      "task_789xyz": {
        "task_id": "task_789xyz",
        "status": "completed",
        "output": {
          "recommendations_count": 2,
          "potential_savings_usd": 73000
        }
      }
    },
    "provenance_hash": "sha256:a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0"
  },
  "metadata": {
    "request_id": "req_abc123",
    "processing_time_ms": 40000
  },
  "timestamp": "2025-12-06T10:30:45Z"
}
```

### Response with Provenance (include_provenance=true)

```json
{
  "success": true,
  "data": {
    "workflow_id": "wf_a1b2c3d4e5f6",
    "final_output": { ... },
    "provenance": {
      "chain_id": "chain_xyz789",
      "merkle_root": "sha256:a1b2c3d4...",
      "record_count": 5,
      "records": [
        {
          "record_id": "rec_001",
          "provenance_type": "INPUT",
          "agent_id": "GL-001",
          "timestamp": "2025-12-06T10:30:05Z",
          "input_hash": "sha256:abc123...",
          "output_hash": "sha256:def456...",
          "formula_id": null
        },
        {
          "record_id": "rec_002",
          "provenance_type": "CALCULATION",
          "agent_id": "GL-002",
          "timestamp": "2025-12-06T10:30:10Z",
          "input_hash": "sha256:ghi789...",
          "output_hash": "sha256:jkl012...",
          "formula_id": "ASME_PTC_4.1",
          "formula_reference": "ASME PTC 4.1-2013"
        }
      ],
      "compliance_frameworks": ["ISO_14064", "GHG_PROTOCOL"]
    }
  },
  "timestamp": "2025-12-06T10:30:45Z"
}
```

### Example (Python)

```python
import requests

headers = {
    "Authorization": f"Bearer {access_token}"
}

response = requests.get(
    f"https://api.greenlang.io/v1/process-heat/results/{workflow_id}",
    headers=headers,
    params={
        "include_provenance": True,
        "include_uncertainty": True
    }
)

result = response.json()
efficiency = result["data"]["final_output"]["efficiency"]
print(f"Net Efficiency: {efficiency['net_efficiency_pct']:.1f}%")

for rec in result["data"]["final_output"]["recommendations"]:
    print(f"- {rec['title']}: Save ${rec['estimated_annual_savings_usd']:,}/year")
```

---

## Cancel Job

Cancel a running or pending workflow job.

```http
DELETE /api/v1/process-heat/jobs/{job_id}
```

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | Workflow job identifier |

### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `force` | boolean | No | Force cancellation even if tasks are running. Default: false |
| `reason` | string | No | Cancellation reason for audit trail |

### Response (200 OK)

```json
{
  "success": true,
  "data": {
    "workflow_id": "wf_a1b2c3d4e5f6",
    "status": "cancelled",
    "cancelled_at": "2025-12-06T10:31:00Z",
    "cancelled_by": "user@example.com",
    "reason": "Operator requested cancellation",
    "tasks_cancelled": 1,
    "tasks_completed": 1
  },
  "timestamp": "2025-12-06T10:31:00Z"
}
```

### Response (409 Conflict - Already Completed)

```json
{
  "success": false,
  "error": {
    "code": "WORKFLOW_ALREADY_COMPLETED",
    "message": "Cannot cancel workflow that has already completed",
    "details": {
      "workflow_id": "wf_a1b2c3d4e5f6",
      "current_status": "completed"
    }
  },
  "timestamp": "2025-12-06T10:31:00Z"
}
```

---

## List Workflows

List all workflows with optional filtering.

```http
GET /api/v1/process-heat/workflows
```

### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `status` | string | No | Filter by status (comma-separated) |
| `workflow_type` | string | No | Filter by workflow type |
| `priority` | string | No | Filter by priority |
| `boiler_id` | string | No | Filter by boiler ID |
| `created_after` | string | No | Filter by creation date (ISO 8601) |
| `created_before` | string | No | Filter by creation date (ISO 8601) |
| `page` | integer | No | Page number. Default: 1 |
| `per_page` | integer | No | Items per page. Default: 20, Max: 100 |

### Response (200 OK)

```json
{
  "success": true,
  "data": {
    "items": [
      {
        "workflow_id": "wf_a1b2c3d4e5f6",
        "name": "Boiler Efficiency Optimization",
        "workflow_type": "optimization",
        "status": "completed",
        "priority": "high",
        "created_at": "2025-12-06T10:30:00Z",
        "completed_at": "2025-12-06T10:30:45Z",
        "duration_ms": 40000
      },
      {
        "workflow_id": "wf_b2c3d4e5f6g7",
        "name": "Daily Emissions Report",
        "workflow_type": "reporting",
        "status": "running",
        "priority": "normal",
        "created_at": "2025-12-06T10:35:00Z",
        "completed_at": null,
        "duration_ms": null
      }
    ],
    "pagination": {
      "page": 1,
      "per_page": 20,
      "total_items": 156,
      "total_pages": 8,
      "has_next": true,
      "has_prev": false
    }
  },
  "timestamp": "2025-12-06T10:40:00Z"
}
```

---

## Agent Management

### List Registered Agents

```http
GET /api/v1/agents
```

### Response (200 OK)

```json
{
  "success": true,
  "data": {
    "agents": [
      {
        "agent_id": "GL-002-001",
        "agent_type": "GL-002",
        "name": "BoilerOptimizer Primary",
        "health": "healthy",
        "version": "2.1.0",
        "last_heartbeat": "2025-12-06T10:39:55Z",
        "active_tasks": 1,
        "completed_tasks": 145,
        "capabilities": [
          "efficiency_calculation",
          "combustion_optimization",
          "steam_analysis",
          "economizer_analysis"
        ]
      },
      {
        "agent_id": "GL-005-001",
        "agent_type": "GL-005",
        "name": "CombustionDiagnostics Primary",
        "health": "healthy",
        "version": "1.8.0",
        "last_heartbeat": "2025-12-06T10:39:58Z",
        "active_tasks": 0,
        "completed_tasks": 89,
        "capabilities": [
          "combustion_analysis",
          "fuel_characterization",
          "anomaly_detection"
        ]
      }
    ]
  },
  "timestamp": "2025-12-06T10:40:00Z"
}
```

### Register Agent

```http
POST /api/v1/agents
```

### Request Body

```json
{
  "agent_id": "GL-002-002",
  "agent_type": "GL-002",
  "name": "BoilerOptimizer Secondary",
  "capabilities": [
    "efficiency_calculation",
    "combustion_optimization"
  ],
  "endpoint": "https://agent-002.internal:8080"
}
```

### Response (201 Created)

```json
{
  "success": true,
  "data": {
    "agent_id": "GL-002-002",
    "registered": true,
    "registered_at": "2025-12-06T10:40:00Z"
  },
  "timestamp": "2025-12-06T10:40:00Z"
}
```

### Get Agent Status

```http
GET /api/v1/agents/{agent_id}
```

### Deregister Agent

```http
DELETE /api/v1/agents/{agent_id}
```

---

## Error Codes

| Code | HTTP Status | Description | Resolution |
|------|-------------|-------------|------------|
| `WORKFLOW_NOT_FOUND` | 404 | Workflow does not exist | Verify workflow ID |
| `WORKFLOW_ALREADY_COMPLETED` | 409 | Cannot modify completed workflow | N/A |
| `WORKFLOW_TIMEOUT` | 408 | Workflow execution timed out | Increase timeout or simplify |
| `AGENT_NOT_AVAILABLE` | 503 | Required agent not available | Check agent health |
| `AGENT_CAPACITY_EXCEEDED` | 429 | Agent at maximum capacity | Wait and retry |
| `TASK_FAILED` | 500 | Task execution failed | Check task error details |
| `INVALID_WORKFLOW_TYPE` | 400 | Invalid workflow type | Use valid workflow type |
| `INVALID_PRIORITY` | 400 | Invalid priority level | Use valid priority |
| `INVALID_PARAMETERS` | 400 | Invalid workflow parameters | Check parameter format |
| `SAFETY_VIOLATION` | 403 | Safety constraint violated | Review safety parameters |
| `DEPENDENCY_CYCLE` | 400 | Task dependency cycle detected | Review task dependencies |
| `CHECKPOINT_FAILED` | 500 | Checkpoint creation failed | Contact support |

---

## Rate Limiting

Orchestrator endpoints have the following rate limits:

| Endpoint | Rate Limit |
|----------|------------|
| `POST /calculate` | 100/minute |
| `GET /status/{job_id}` | 300/minute |
| `GET /results/{job_id}` | 100/minute |
| `DELETE /jobs/{job_id}` | 50/minute |
| `GET /workflows` | 200/minute |

---

## Best Practices

1. **Use idempotency keys** for `POST /calculate` to safely retry requests
2. **Poll status with exponential backoff** rather than rapid polling
3. **Use webhooks** for real-time completion notifications
4. **Specify priorities appropriately** - reserve `critical` and `emergency` for urgent cases
5. **Enable checkpointing** for long-running workflows to support recovery
6. **Include metadata** for easier tracking and debugging
7. **Handle all error codes** in your integration

---

## See Also

- [Main API Reference](../process_heat_api_reference.md)
- [Emissions API](emissions.md)
- [Compliance API](compliance.md)
- [Common Data Models](../schemas/common_models.md)
