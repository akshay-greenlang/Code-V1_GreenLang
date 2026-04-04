# Threat Modeling Service API Reference (SEC-010)

## Overview

The Threat Modeling Service provides STRIDE-based threat analysis, data flow diagram management, risk scoring, and mitigation tracking. Supports creating threat models for services, adding components and data flows, running automated analysis, and generating reports.

**Router Prefix:** `/api/v1/secops/threats`
**Tags:** `Threat Modeling`
**Source:** `greenlang/infrastructure/threat_modeling/api/threat_routes.py`

---

## Endpoint Summary

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/api/v1/secops/threats` | List threat models | Yes |
| POST | `/api/v1/secops/threats` | Create threat model | Yes |
| GET | `/api/v1/secops/threats/{id}` | Get threat model details | Yes |
| PUT | `/api/v1/secops/threats/{id}` | Update threat model | Yes |
| DELETE | `/api/v1/secops/threats/{id}` | Delete threat model (draft only) | Yes |
| POST | `/api/v1/secops/threats/{id}/analyze` | Run STRIDE analysis | Yes |
| POST | `/api/v1/secops/threats/{id}/components` | Add component | Yes |
| PUT | `/api/v1/secops/threats/{id}/components/{cid}` | Update component | Yes |
| POST | `/api/v1/secops/threats/{id}/data-flows` | Add data flow | Yes |
| POST | `/api/v1/secops/threats/{id}/mitigations` | Add mitigation | Yes |
| PUT | `/api/v1/secops/threats/{id}/approve` | Approve threat model | Yes |
| GET | `/api/v1/secops/threats/{id}/report` | Generate report | Yes |

**Permissions:**
- Read operations: `secops:threats:read`
- Write operations: `secops:threats:write`

---

## Endpoints

### GET /api/v1/secops/threats

List all threat models with pagination and filtering.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | integer | 1 | Page number |
| `page_size` | integer | 20 | Items per page (max 100) |
| `status` | string | - | Filter: `draft`, `in_review`, `approved`, `archived` |
| `service_name` | string | - | Filter by service name (partial match) |

**Response (200 OK):**

```json
{
  "total": 8,
  "page": 1,
  "page_size": 20,
  "items": [
    {
      "id": "tm-uuid-001",
      "service_name": "cbam-intake-service",
      "version": "1.2.0",
      "status": "approved",
      "overall_risk_score": 6.2,
      "threat_count": 15,
      "mitigated_count": 12,
      "critical_count": 0,
      "high_count": 3,
      "owner": "platform-team",
      "created_at": "2026-02-15T10:00:00Z",
      "updated_at": "2026-03-20T14:30:00Z",
      "approved_by": "security-lead",
      "approved_at": "2026-03-20T16:00:00Z"
    }
  ]
}
```

---

### POST /api/v1/secops/threats

Create a new threat model. Service name is normalized to lowercase with hyphens.

**Request Body:**

```json
{
  "service_name": "CBAM Intake Service",
  "version": "1.0.0",
  "description": "Service handling CBAM data intake from importers, including file upload, validation, and processing.",
  "scope": "Covers the intake API, file processing pipeline, and integration with the calculation engine.",
  "owner": "platform-team",
  "tags": ["cbam", "data-intake", "file-upload"]
}
```

**Response (201 Created):** Returns `ThreatModelResponse` with initial `draft` status and zero risk scores.

**Error Responses:**

| Status Code | Description |
|-------------|-------------|
| 409 | Conflict - Active threat model for this service already exists |

---

### POST /api/v1/secops/threats/{id}/analyze

Run STRIDE analysis on the threat model. Analyzes all components and data flows to identify threats, calculate risk scores, and optionally validate the data flow diagram.

**Request Body:**

```json
{
  "include_generic_threats": true,
  "recalculate_scores": true,
  "validate_dfd": true
}
```

**Response (200 OK):**

```json
{
  "threat_model_id": "tm-uuid-001",
  "threats_identified": 15,
  "threats_by_category": {
    "spoofing": 3,
    "tampering": 2,
    "repudiation": 1,
    "information_disclosure": 4,
    "denial_of_service": 3,
    "elevation_of_privilege": 2
  },
  "threats_by_severity": {
    "critical": 0,
    "high": 3,
    "medium": 8,
    "low": 4
  },
  "overall_risk_score": 6.2,
  "validation_passed": true,
  "validation_errors": 0,
  "validation_warnings": 1,
  "analysis_duration_ms": 245.5
}
```

---

### POST /api/v1/secops/threats/{id}/components

Add a component to the threat model's data flow diagram.

**Request Body:**

```json
{
  "name": "CBAM API Gateway",
  "component_type": "web_server",
  "description": "FastAPI application serving the CBAM intake REST API",
  "trust_level": 3,
  "data_classification": "confidential",
  "owner": "platform-team",
  "technology_stack": ["python", "fastapi", "uvicorn"],
  "is_external": false,
  "network_zone": "dmz"
}
```

**Component Types:** `web_server`, `database`, `message_queue`, `api_gateway`, `cache`, `storage`, `external_service`, `process`, `user`, `unknown`

**Data Classifications:** `public`, `internal`, `confidential`, `restricted`

---

### POST /api/v1/secops/threats/{id}/data-flows

Add a data flow between two components. Validates that both component IDs exist in the model.

**Request Body:**

```json
{
  "source_component_id": "comp-api-gateway",
  "destination_component_id": "comp-database",
  "data_type": "cbam_shipment_data",
  "protocol": "postgresql",
  "encryption": true,
  "authentication_required": true,
  "authentication_method": "certificate",
  "authorization_required": true,
  "description": "CBAM shipment data written to PostgreSQL",
  "data_classification": "confidential",
  "bidirectional": true
}
```

---

### POST /api/v1/secops/threats/{id}/mitigations

Add a mitigation for an identified threat. Validates that the threat ID exists.

**Request Body:**

```json
{
  "threat_id": "threat-uuid-001",
  "control_id": "SEC-002",
  "title": "Implement RBAC on report download endpoint",
  "description": "Add tenant-scoped authorization check to verify the requesting user belongs to the organization that owns the report.",
  "effectiveness": 95.0,
  "owner": "platform-team",
  "priority": 1,
  "due_date": "2026-05-01T00:00:00Z"
}
```

---

### PUT /api/v1/secops/threats/{id}/approve

Approve a threat model. Cannot approve if there are critical unmitigated threats.

**Request Body:**

```json
{
  "approved_by": "security-lead",
  "comments": "All critical and high threats have been addressed. Remaining medium threats have accepted risk with compensating controls."
}
```

**Error Responses:**

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request - Cannot approve with unmitigated critical threats |
| 400 | Bad Request - Cannot approve threat model in current status |

---

### GET /api/v1/secops/threats/{id}/report

Generate a threat model report in the specified format.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | string | `json` | Report format: `json`, `pdf`, `html`, `markdown` |

**Response (200 OK):**

```json
{
  "threat_model_id": "tm-uuid-001",
  "service_name": "cbam-intake-service",
  "format": "json",
  "content": "{...}",
  "download_url": null,
  "generated_at": "2026-04-04T12:00:00Z"
}
```

For PDF/HTML formats, `content` is null and a `download_url` is provided.

---

## STRIDE Categories

| Category | Code | Description |
|----------|------|-------------|
| Spoofing | S | Impersonating a user or system |
| Tampering | T | Unauthorized modification of data |
| Repudiation | R | Denying an action without proof |
| Information Disclosure | I | Exposing data to unauthorized parties |
| Denial of Service | D | Making a service unavailable |
| Elevation of Privilege | E | Gaining unauthorized access |

## Threat Model Lifecycle

```
draft -> in_review -> approved -> archived
```

- Only `draft` models can be deleted
- Approved models cannot be updated (create a new version instead)
- Critical unmitigated threats block approval
