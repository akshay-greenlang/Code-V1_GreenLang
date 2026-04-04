# Incident Response Service API Reference (SEC-010)

## Overview

The Incident Response Service provides security incident lifecycle management, including creation, acknowledgment, assignment, playbook execution, resolution, and closure. Includes timeline tracking and MTTD/MTTR metrics for operational excellence.

**Router Prefix:** `/api/v1/secops/incidents`
**Tags:** `Incident Response`
**Source:** `greenlang/infrastructure/incident_response/api/incident_routes.py`

---

## Endpoint Summary

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/api/v1/secops/incidents` | List incidents with pagination/filters | Yes |
| GET | `/api/v1/secops/incidents/{id}` | Get incident details | Yes |
| POST | `/api/v1/secops/incidents/{id}/acknowledge` | Acknowledge incident | Yes |
| POST | `/api/v1/secops/incidents/{id}/assign` | Assign responder | Yes |
| POST | `/api/v1/secops/incidents/{id}/execute-playbook` | Execute playbook | Yes |
| PUT | `/api/v1/secops/incidents/{id}/resolve` | Resolve incident | Yes |
| PUT | `/api/v1/secops/incidents/{id}/close` | Close incident | Yes |
| GET | `/api/v1/secops/incidents/{id}/timeline` | Get incident timeline | Yes |
| GET | `/api/v1/secops/incidents/metrics` | MTTD/MTTR metrics | Yes |

---

## Endpoints

### GET /api/v1/secops/incidents

List incidents with pagination and optional filtering by severity, status, and assignee.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `severity` | string | - | Filter by severity: `critical`, `high`, `medium`, `low` |
| `status` | string | - | Filter by status: `new`, `acknowledged`, `investigating`, `resolved`, `closed` |
| `assignee` | string | - | Filter by assigned responder |
| `page` | integer | 1 | Page number |
| `page_size` | integer | 20 | Items per page |

---

### POST /api/v1/secops/incidents/{id}/acknowledge

Acknowledge receipt of an incident. Transitions status from `new` to `acknowledged`. Records the acknowledging user and timestamp.

---

### POST /api/v1/secops/incidents/{id}/execute-playbook

Execute an automated playbook against an incident. Playbooks define automated response actions (e.g., isolate host, block IP, rotate credentials).

**Request Body:**

```json
{
  "playbook_id": "pb-isolate-host",
  "parameters": {
    "host_id": "srv-web-03",
    "duration_hours": 24
  }
}
```

---

### GET /api/v1/secops/incidents/metrics

Get MTTD (Mean Time to Detect) and MTTR (Mean Time to Respond) metrics.

**Response (200 OK):**

```json
{
  "total_incidents": 45,
  "open_incidents": 3,
  "mttd_hours": 0.5,
  "mttr_hours": 4.2,
  "incidents_by_severity": {
    "critical": 2,
    "high": 8,
    "medium": 20,
    "low": 15
  },
  "resolution_rate_pct": 93.3,
  "period": "last_90_days"
}
```

---

## Incident Lifecycle

```
new -> acknowledged -> investigating -> resolved -> closed
```

Each transition is recorded in the incident timeline with actor, timestamp, and notes.
