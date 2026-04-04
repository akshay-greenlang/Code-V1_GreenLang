# GreenLang Foundation Agents -- API Reference

This document provides an overview of the 10 Foundation agents that form the core
infrastructure layer of the GreenLang platform. Each agent exposes a REST API
built on FastAPI and follows common conventions for authentication, error
handling, and provenance tracking.

---

## Agents at a Glance

| # | Agent ID | Name | Base Path | Endpoints | Description |
|---|----------|------|-----------|-----------|-------------|
| 1 | AGENT-FOUND-001 | [DAG Orchestrator](orchestrator.md) | `/api/v1/orchestrator` | 20+ (DAG) + 15+ (Control Plane) | DAG workflow management, pipeline execution, approvals |
| 2 | AGENT-FOUND-002 | [Schema Compiler & Validator](schema_compiler.md) | `/v1/schema` | 8 | Schema validation, batch validation, compilation, registry |
| 3 | AGENT-FOUND-003 | [Unit & Reference Normalizer](unit_normalizer.md) | `/api/v1/normalizer` | 15 | Unit conversion, GHG conversion, entity resolution, GWP lookup |
| 4 | AGENT-FOUND-004 | [Assumptions Registry](assumptions_registry.md) | `/api/v1/assumptions` | 20 | Assumption CRUD, scenarios, validation, sensitivity analysis |
| 5 | AGENT-FOUND-005 | [Citations & Evidence](citations_evidence.md) | `/api/v1/citations` | 20 | Citation management, evidence packages, verification, export/import |
| 6 | AGENT-FOUND-006 | [Access & Policy Guard](access_policy.md) | `/api/v1/access-guard` | 20 | Access control, policy management, OPA integration, audit logging |
| 7 | AGENT-FOUND-007 | [Agent Registry & Service Catalog](agent_registry.md) | `/api/v1/agent-registry` | 20 | Agent registration, discovery, health, dependency resolution |
| 8 | AGENT-FOUND-008 | [Reproducibility Agent](reproducibility.md) | `/api/v1/reproducibility` | 20 | Verification, drift detection, replay, environment capture |
| 9 | AGENT-FOUND-009 | [QA Test Harness](qa_harness.md) | `/api/v1/qa-test-harness` | 20 | Test execution, golden files, benchmarks, coverage, reporting |
| 10 | AGENT-FOUND-010 | [Observability & Telemetry](observability.md) | `/api/v1/observability-agent` | 20 | Metrics, traces, logs, alerts, SLOs, dashboards |

---

## Common Conventions

### Authentication

All endpoints require a valid JWT Bearer token or API key unless noted otherwise.
Health-check endpoints (`/health`) are unauthenticated by default.

```
Authorization: Bearer <token>
```

or

```
X-API-Key: <api_key>
```

### Request Tracing

Every request generates or propagates a trace ID via the `X-Trace-ID` header.
The trace ID appears in all error responses and can be used for distributed
tracing correlation.

### Error Response Format

All agents return errors in a standard envelope:

```json
{
  "error": "validation_error",
  "message": "Human-readable description",
  "details": [
    {
      "code": "GL-E-VAL-002",
      "message": "Specific validation detail",
      "field": "optional_field_name"
    }
  ],
  "trace_id": "tr-abc123",
  "timestamp": "2026-04-04T12:00:00Z"
}
```

### Common HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Resource created |
| 202 | Accepted (async processing) |
| 204 | No content (successful delete) |
| 400 | Bad request / validation error |
| 401 | Unauthorized (missing or invalid credentials) |
| 403 | Forbidden (policy violation) |
| 404 | Resource not found |
| 409 | Conflict (e.g., resource already exists, already decided) |
| 413 | Payload too large (batch size exceeded) |
| 429 | Rate limit exceeded |
| 500 | Internal server error |
| 503 | Service unavailable (agent not configured) |

### Provenance

Many mutation endpoints return a `provenance_hash` field -- a SHA-256 hash
capturing the state change for audit trail purposes.

---

## Source Code Locations

All foundation agent routers are located under:

```
greenlang/agents/foundation/
  orchestrator/api/   -- dag_router.py, routes.py, approval_routes.py
  schema/api/         -- routes.py
  normalizer/api/     -- router.py
  assumptions/api/    -- router.py
  citations/api/      -- router.py
  access_guard/api/   -- router.py
  agent_registry/api/ -- router.py
  reproducibility/api/ -- router.py
  qa_test_harness/api/ -- router.py
  observability_agent/api/ -- router.py
```
