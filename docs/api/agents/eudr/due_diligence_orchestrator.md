# AGENT-EUDR-026: Due Diligence Orchestrator API

**Agent ID:** `GL-EUDR-DDO-026`
**Prefix:** `/api/v1/eudr-ddo`
**Version:** 1.0.0
**PRD:** AGENT-EUDR-026
**Regulation:** EU 2023/1115 (EUDR) -- Due diligence workflow per Articles 8-12

## Purpose

The Due Diligence Orchestrator agent coordinates the end-to-end EUDR due
diligence workflow across all other EUDR agents. It manages workflow
templates, executes multi-step workflows (information gathering, risk
assessment, risk mitigation, DDS creation, submission), enforces quality
gates between phases, supports pause/resume/cancel/rollback of workflows,
tracks progress and ETAs, manages checkpoints for recovery, and provides
monitoring dashboards.

---

## Endpoint Summary

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| POST | `/workflows` | Create workflow | JWT |
| GET | `/workflows` | List workflows | JWT |
| GET | `/workflows/{workflow_id}` | Get workflow details | JWT |
| PUT | `/workflows/{workflow_id}` | Update workflow | JWT |
| DELETE | `/workflows/{workflow_id}` | Delete workflow | JWT |
| POST | `/workflows/{workflow_id}/clone` | Clone workflow | JWT |
| POST | `/workflows/{workflow_id}/start` | Start execution | JWT |
| POST | `/workflows/{workflow_id}/pause` | Pause execution | JWT |
| POST | `/workflows/{workflow_id}/resume` | Resume execution | JWT |
| POST | `/workflows/{workflow_id}/cancel` | Cancel execution | JWT |
| POST | `/workflows/{workflow_id}/rollback` | Rollback to checkpoint | JWT |
| GET | `/workflows/{workflow_id}/status` | Get execution status | JWT |
| GET | `/workflows/{workflow_id}/progress` | Get progress details | JWT |
| GET | `/workflows/{workflow_id}/phase-status` | Get per-phase status | JWT |
| GET | `/workflows/{workflow_id}/eta` | Get completion ETA | JWT |
| GET | `/workflows/{workflow_id}/gates` | List quality gates | JWT |
| POST | `/workflows/{workflow_id}/gates/{gate_id}/override` | Override gate | JWT |
| GET | `/workflows/{workflow_id}/gates/{gate_id}` | Get gate details | JWT |
| GET | `/workflows/{workflow_id}/checkpoints` | List checkpoints | JWT |
| POST | `/workflows/{workflow_id}/checkpoints` | Create checkpoint | JWT |
| GET | `/workflows/{workflow_id}/checkpoints/{cp_id}` | Get checkpoint | JWT |
| GET | `/templates` | List workflow templates | JWT |
| GET | `/templates/{template_id}` | Get template details | JWT |
| POST | `/templates` | Create template | JWT |
| PUT | `/templates/{template_id}` | Update template | JWT |
| POST | `/packages/{workflow_id}/generate` | Generate DD package | JWT |
| GET | `/packages/{workflow_id}` | Get DD package | JWT |
| GET | `/packages/{workflow_id}/download` | Download package | JWT |
| POST | `/packages/{workflow_id}/validate` | Validate package | JWT |
| GET | `/monitoring/health` | Get system health | JWT |
| GET | `/monitoring/metrics` | Get system metrics | JWT |
| GET | `/monitoring/version` | Get version info | JWT |
| GET | `/monitoring/circuit-breakers` | Get circuit breaker states | JWT |
| GET | `/monitoring/dlq` | Get dead letter queue | JWT |
| POST | `/monitoring/dlq/retry` | Retry DLQ message | JWT |

**Total: 35 endpoints**

---

## Endpoints

### POST /api/v1/eudr-ddo/workflows

Create a new due diligence workflow for an operator and commodity.

**Request:**

```json
{
  "operator_id": "OP-2024-001",
  "commodity": "cocoa",
  "template_id": "tpl_eudr_standard",
  "name": "Ghana Cocoa DD Q1 2026",
  "sourcing_countries": ["GH", "CI"],
  "supplier_ids": ["sup-001", "sup-002", "sup-003"],
  "priority": "high",
  "due_date": "2026-06-30",
  "phases": [
    "information_gathering",
    "risk_assessment",
    "risk_mitigation",
    "dds_creation",
    "submission"
  ]
}
```

**Response (201 Created):**

```json
{
  "workflow_id": "wf_001",
  "operator_id": "OP-2024-001",
  "commodity": "cocoa",
  "template_id": "tpl_eudr_standard",
  "name": "Ghana Cocoa DD Q1 2026",
  "status": "created",
  "phase_count": 5,
  "current_phase": null,
  "quality_gates": 4,
  "due_date": "2026-06-30",
  "created_at": "2026-04-04T10:00:00Z"
}
```

---

### POST /api/v1/eudr-ddo/workflows/{workflow_id}/start

Start execution of a due diligence workflow. The orchestrator begins the first
phase and coordinates agent calls across the platform.

**Request:**

```json
{
  "execution_mode": "automatic",
  "notify_on_gate": true,
  "checkpoint_interval_minutes": 30
}
```

**Response (200 OK):**

```json
{
  "workflow_id": "wf_001",
  "status": "running",
  "current_phase": "information_gathering",
  "started_at": "2026-04-04T10:05:00Z",
  "estimated_completion": "2026-04-04T14:00:00Z",
  "agents_invoked": [
    "GL-EUDR-IGA-027",
    "GL-EUDR-SCM-001",
    "GL-EUDR-GEO-002"
  ]
}
```

---

### GET /api/v1/eudr-ddo/workflows/{workflow_id}/progress

Get detailed progress for each phase of a running workflow, including
per-step status and blocking issues.

**Response (200 OK):**

```json
{
  "workflow_id": "wf_001",
  "overall_progress_pct": 42.0,
  "phases": [
    {
      "phase": "information_gathering",
      "status": "completed",
      "progress_pct": 100.0,
      "started_at": "2026-04-04T10:05:00Z",
      "completed_at": "2026-04-04T11:15:00Z",
      "steps_total": 8,
      "steps_completed": 8
    },
    {
      "phase": "risk_assessment",
      "status": "running",
      "progress_pct": 60.0,
      "started_at": "2026-04-04T11:16:00Z",
      "steps_total": 5,
      "steps_completed": 3,
      "current_step": "country_risk_evaluation",
      "blocking_issues": []
    },
    {
      "phase": "risk_mitigation",
      "status": "pending",
      "progress_pct": 0.0
    }
  ],
  "eta": "2026-04-04T14:00:00Z"
}
```

---

### POST /api/v1/eudr-ddo/workflows/{workflow_id}/gates/{gate_id}/override

Override a quality gate that is blocking workflow progression. Requires
elevated permissions and records justification in the audit trail.

**Request:**

```json
{
  "justification": "Supplier documentation will arrive within 48 hours; proceeding to avoid deadline",
  "approved_by": "compliance-manager@company.com",
  "conditions": ["documentation_to_follow_within_48h"]
}
```

**Response (200 OK):**

```json
{
  "gate_id": "gate_002",
  "workflow_id": "wf_001",
  "previous_status": "blocked",
  "new_status": "overridden",
  "overridden_by": "compliance-manager@company.com",
  "justification": "Supplier documentation will arrive within 48 hours",
  "conditions": ["documentation_to_follow_within_48h"],
  "audit_trail_id": "at_001",
  "overridden_at": "2026-04-04T12:00:00Z"
}
```

---

## Error Responses

| Status | Error Code | Description |
|--------|------------|-------------|
| 400 | `invalid_workflow` | Workflow configuration is invalid |
| 404 | `workflow_not_found` | Workflow ID not found |
| 409 | `invalid_state_transition` | Workflow cannot transition to requested state |
| 422 | `gate_not_blockable` | Quality gate is not in a blockable state |
| 503 | `agent_unavailable` | Downstream agent is unreachable |
