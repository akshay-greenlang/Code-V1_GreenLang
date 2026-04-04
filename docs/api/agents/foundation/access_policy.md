# AGENT-FOUND-006: Access & Policy Guard API Reference

**Agent ID:** AGENT-FOUND-006
**Service:** Access & Policy Guard
**Status:** Production Ready
**Base Path:** `/api/v1/access-guard`
**Tag:** `access-guard`
**Source:** `greenlang/agents/foundation/access_guard/api/router.py`

The Access & Policy Guard provides endpoints for access control evaluation,
policy management, resource classification, rate limiting, audit logging, OPA
(Open Policy Agent) integration, policy simulation, and compliance reporting.

---

## Endpoint Summary

| # | Method | Path | Summary | Status Codes |
|---|--------|------|---------|--------------|
| 1 | GET | `/health` | Health check | 200, 503 |
| 2 | POST | `/check` | Check access | 200, 400, 503 |
| 3 | GET | `/policies` | List policies | 200, 503 |
| 4 | POST | `/policies` | Add policy | 201, 400, 503 |
| 5 | GET | `/policies/{policy_id}` | Get policy | 200, 404, 503 |
| 6 | PUT | `/policies/{policy_id}` | Update policy | 200, 400, 404, 503 |
| 7 | DELETE | `/policies/{policy_id}` | Delete policy | 200, 404, 503 |
| 8 | GET | `/rules/effective` | Get effective rules | 200, 503 |
| 9 | POST | `/classify` | Classify resource | 200, 400, 503 |
| 10 | GET | `/rate-limits/{tenant_id}/{principal_id}` | Get rate limit quota | 200, 503 |
| 11 | DELETE | `/rate-limits/{tenant_id}/{principal_id}` | Reset rate limits | 200, 503 |
| 12 | GET | `/audit/events` | Get audit events | 200, 503 |
| 13 | GET | `/audit/events/{event_id}` | Get audit event by ID | 200, 404, 503 |
| 14 | POST | `/audit/compliance-report` | Generate compliance report | 200, 400, 503 |
| 15 | POST | `/opa/policies` | Add Rego policy | 201, 400, 503 |
| 16 | GET | `/opa/policies` | List Rego policies | 200, 503 |
| 17 | DELETE | `/opa/policies/{policy_id}` | Delete Rego policy | 200, 404, 503 |
| 18 | POST | `/simulate` | Simulate policy evaluation | 200, 400, 503 |
| 19 | GET | `/provenance/{entity_id}` | Get provenance chain | 200, 503 |
| 20 | GET | `/metrics` | Get metrics summary | 200, 503 |

---

## Detailed Endpoints

### POST /check -- Check Access

Evaluate an access request against loaded policies using RBAC + ABAC rules.

**Request Body:**

```json
{
  "principal": {
    "principal_id": "user_456",
    "tenant_id": "tenant_001",
    "roles": ["compliance_officer"],
    "attributes": {
      "department": "sustainability",
      "clearance_level": "confidential"
    }
  },
  "resource": {
    "resource_id": "report_2025q4",
    "resource_type": "emissions_report",
    "tenant_id": "tenant_001",
    "classification": "confidential",
    "attributes": {
      "region": "EU"
    }
  },
  "action": "read",
  "context": {
    "request_time": "2026-04-04T10:00:00Z",
    "ip_address": "10.0.1.50"
  },
  "source_ip": "10.0.1.50"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `principal` | object | Yes | Principal data (user/service identity) |
| `resource` | object | Yes | Resource being accessed |
| `action` | string | Yes | Action to check (`read`, `write`, `delete`, `execute`) |
| `context` | object | No | Additional ABAC context |
| `source_ip` | string | No | Source IP address |

**Response (200):**

```json
{
  "allowed": true,
  "decision_reason": "Role 'compliance_officer' has 'read' access to 'emissions_report'",
  "matched_policies": ["pol_compliance_read"],
  "evaluation_time_ms": 1.2,
  "provenance_hash": "sha256:..."
}
```

---

### POST /policies -- Add Policy

**Request Body:**

```json
{
  "policy_id": "pol_compliance_read",
  "name": "Compliance Officer Read Access",
  "description": "Allow compliance officers to read all reports",
  "version": "1.0.0",
  "enabled": true,
  "rules": [
    {
      "effect": "allow",
      "principals": { "roles": ["compliance_officer"] },
      "actions": ["read"],
      "resources": { "types": ["emissions_report", "compliance_report"] }
    }
  ],
  "tenant_id": "tenant_001",
  "applies_to": ["emissions_report", "compliance_report"],
  "created_by": "admin@company.com"
}
```

**Response (201):**

```json
{
  "policy_id": "pol_compliance_read",
  "provenance_hash": "sha256:..."
}
```

---

### POST /classify -- Classify Resource

Classify a resource by sensitivity level for data loss prevention.

**Request Body:**

```json
{
  "resource_id": "report_2025q4",
  "resource_type": "emissions_report",
  "tenant_id": "tenant_001",
  "classification": "internal",
  "attributes": {
    "contains_pii": false,
    "data_origin": "supplier_declarations"
  }
}
```

**Response (200):**

```json
{
  "resource_id": "report_2025q4",
  "classification": "confidential"
}
```

---

### GET /rate-limits/{tenant_id}/{principal_id} -- Get Rate Limit Quota

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `role` | string | Principal role for tier-based limits |

**Response (200):**

```json
{
  "tenant_id": "tenant_001",
  "principal_id": "user_456",
  "remaining_requests": 85,
  "limit": 100,
  "window_seconds": 60,
  "resets_at": "2026-04-04T10:01:00Z"
}
```

---

### GET /audit/events -- Get Audit Events

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tenant_id` | string | -- | Filter by tenant |
| `event_type` | string | -- | Filter by event type |
| `limit` | integer | 100 | Max results (1-1000) |
| `offset` | integer | 0 | Offset |

**Response (200):** Array of audit event objects.

---

### POST /audit/compliance-report -- Generate Compliance Report

**Request Body:**

```json
{
  "tenant_id": "tenant_001",
  "start_date": "2026-01-01T00:00:00Z",
  "end_date": "2026-03-31T23:59:59Z"
}
```

**Response (200):**

```json
{
  "tenant_id": "tenant_001",
  "period_start": "2026-01-01T00:00:00Z",
  "period_end": "2026-03-31T23:59:59Z",
  "total_access_requests": 15230,
  "allowed_count": 15100,
  "denied_count": 130,
  "policy_violations": 12,
  "unique_principals": 45,
  "resources_accessed": 320
}
```

---

### POST /opa/policies -- Add Rego Policy

Add an Open Policy Agent (OPA) Rego policy for advanced policy evaluation.

**Request Body:**

```json
{
  "policy_id": "rego_geo_restriction",
  "rego_source": "package greenlang.access\n\ndefault allow = false\n\nallow {\n  input.resource.region == input.principal.allowed_region\n}"
}
```

**Response (201):**

```json
{
  "policy_id": "rego_geo_restriction",
  "hash": "sha256:..."
}
```

---

### POST /simulate -- Simulate Policy Evaluation

Run a policy simulation against test requests without affecting real access
control decisions.

**Request Body:**

```json
{
  "requests": [
    {
      "principal": { "principal_id": "user_789", "roles": ["viewer"] },
      "resource": { "resource_id": "report_2025q4", "resource_type": "emissions_report" },
      "action": "delete"
    }
  ],
  "policy_ids": ["pol_compliance_read"]
}
```

**Response (200):**

```json
{
  "test_requests": 1,
  "results": [
    {
      "allowed": false,
      "decision_reason": "Role 'viewer' does not have 'delete' access"
    }
  ]
}
```
