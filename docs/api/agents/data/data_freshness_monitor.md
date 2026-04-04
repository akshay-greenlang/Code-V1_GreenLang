# Data Freshness Monitor API Reference

**Agent:** AGENT-DATA-016 (GL-DATA-X-019)
**Prefix:** `/api/v1/freshness`
**Source:** `greenlang/agents/data/data_freshness_monitor/api/router.py`
**Status:** Production Ready

## Overview

The Data Freshness Monitor agent provides 20 REST API endpoints for monitoring the timeliness of data feeds critical to compliance reporting. Capabilities include dataset registration with refresh cadence expectations, SLA definition with warning and critical thresholds, freshness checking (single and batch), SLA breach detection and lifecycle management, alert generation, machine-learning-based refresh prediction, and full monitoring pipeline orchestration.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/datasets` | Register dataset for monitoring | Yes |
| 2 | GET | `/datasets` | List registered datasets | Yes |
| 3 | GET | `/datasets/{dataset_id}` | Get dataset details | Yes |
| 4 | PUT | `/datasets/{dataset_id}` | Update dataset metadata | Yes |
| 5 | DELETE | `/datasets/{dataset_id}` | Remove dataset from monitoring | Yes |
| 6 | POST | `/sla` | Create SLA definition | Yes |
| 7 | GET | `/sla` | List SLA definitions | Yes |
| 8 | GET | `/sla/{sla_id}` | Get SLA details | Yes |
| 9 | PUT | `/sla/{sla_id}` | Update SLA definition | Yes |
| 10 | POST | `/check` | Run freshness check (single dataset) | Yes |
| 11 | POST | `/check/batch` | Run batch freshness check | Yes |
| 12 | GET | `/checks` | List check results | Yes |
| 13 | GET | `/breaches` | List SLA breaches | Yes |
| 14 | GET | `/breaches/{breach_id}` | Get breach details | Yes |
| 15 | PUT | `/breaches/{breach_id}` | Update breach status | Yes |
| 16 | GET | `/alerts` | List alerts | Yes |
| 17 | GET | `/predictions` | Get refresh predictions | Yes |
| 18 | POST | `/pipeline` | Run full monitoring pipeline | Yes |
| 19 | GET | `/health` | Health check | No |
| 20 | GET | `/stats` | Service statistics | Yes |

---

## Key Endpoints

### 1. Register Dataset

Register a dataset for freshness monitoring, specifying its expected refresh cadence and priority.

```http
POST /api/v1/freshness/datasets
```

**Request Body:**

```json
{
  "name": "Utility Bills - Natural Gas",
  "source": "erp_sap",
  "owner": "sustainability-team",
  "refresh_cadence": "monthly",
  "priority": 2,
  "tags": ["energy", "scope2", "natural_gas"],
  "metadata": {"erp_module": "MM", "cost_center": "CC-4200"}
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Human-readable dataset name |
| `source` | string | No | Source system identifier |
| `owner` | string | No | Responsible team or individual |
| `refresh_cadence` | string | No | Expected frequency: `realtime`, `hourly`, `daily`, `weekly`, `monthly`, `quarterly`, `annual` (default: `daily`) |
| `priority` | integer | No | Priority 1 (highest) to 10 (lowest), default 5 |
| `tags` | array | No | Tags for grouping and filtering |
| `metadata` | object | No | Additional dataset metadata |

**Response (201):**

```json
{
  "status": "created",
  "data": {
    "dataset_id": "ds_abc123",
    "name": "Utility Bills - Natural Gas",
    "source": "erp_sap",
    "owner": "sustainability-team",
    "refresh_cadence": "monthly",
    "priority": 2,
    "tags": ["energy", "scope2", "natural_gas"],
    "status": "active",
    "last_refreshed_at": null,
    "freshness_score": null,
    "sla_status": "unknown",
    "created_at": "2026-04-04T10:30:00Z"
  }
}
```

---

### 6. Create SLA Definition

Define an SLA with warning and critical thresholds for a dataset.

```http
POST /api/v1/freshness/sla
```

**Request Body:**

```json
{
  "dataset_id": "ds_abc123",
  "name": "Monthly utility data SLA",
  "warning_hours": 48.0,
  "critical_hours": 168.0,
  "severity": "high",
  "escalation_policy": {
    "warning": {"notify": ["sustainability-team@example.com"]},
    "critical": {"notify": ["sustainability-team@example.com", "cfo@example.com"]}
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `dataset_id` | string | No | Dataset this SLA applies to (empty for default) |
| `name` | string | No | SLA display name |
| `warning_hours` | float | No | Hours before warning alert (default: 24.0) |
| `critical_hours` | float | No | Hours before critical alert (default: 72.0) |
| `severity` | string | No | Breach severity: `info`, `low`, `medium`, `high`, `critical` (default: `high`) |
| `escalation_policy` | object | No | Escalation chain configuration |

---

### 10. Run Freshness Check

Run a freshness check on a single dataset, comparing its last refresh timestamp against SLA thresholds.

```http
POST /api/v1/freshness/check
```

**Request Body:**

```json
{
  "dataset_id": "ds_abc123",
  "last_refreshed_at": "2026-03-15T08:00:00Z"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `dataset_id` | string | Yes | Dataset to check |
| `last_refreshed_at` | string | No | ISO timestamp of last refresh (uses stored metadata if omitted) |

**Response (200):**

```json
{
  "status": "ok",
  "data": {
    "check_id": "chk_abc123",
    "dataset_id": "ds_abc123",
    "checked_at": "2026-04-04T10:30:00Z",
    "age_hours": 482.5,
    "freshness_score": 0.15,
    "freshness_level": "stale",
    "sla_status": "critical",
    "processing_time_ms": 8.2,
    "provenance_hash": "sha256:..."
  }
}
```

---

### 13. List SLA Breaches

List SLA breaches with optional severity and status filtering.

```http
GET /api/v1/freshness/breaches?severity=critical&status=detected&limit=50&offset=0
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `severity` | string | - | Filter: `info`, `low`, `medium`, `high`, `critical`, `warning` |
| `status` | string | - | Filter: `detected`, `acknowledged`, `investigating`, `resolved` |
| `limit` | integer | 50 | Max results (1-1000) |
| `offset` | integer | 0 | Pagination offset |

---

### 15. Update Breach Status

Advance a breach through its lifecycle (detected, acknowledged, investigating, resolved).

```http
PUT /api/v1/freshness/breaches/{breach_id}
```

**Request Body:**

```json
{
  "status": "resolved",
  "resolution_notes": "Data feed restored after ERP maintenance window. Confirmed complete dataset received."
}
```

---

### 18. Run Full Monitoring Pipeline

Run the complete monitoring pipeline: check all datasets, detect breaches, generate alerts, and run refresh predictions.

```http
POST /api/v1/freshness/pipeline
```

**Request Body:**

```json
{
  "dataset_ids": ["ds_abc123", "ds_def456"],
  "run_predictions": true,
  "generate_alerts": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `dataset_ids` | array | No | Datasets to monitor (all if omitted) |
| `run_predictions` | boolean | No | Run refresh predictions (default: `true`) |
| `generate_alerts` | boolean | No | Generate alerts for breaches (default: `true`) |

**Response (200):** Pipeline result including datasets checked, breaches detected, alerts generated, predictions made, and provenance hash.

---

## Error Responses

All responses are wrapped in the envelope:

```json
{
  "status": "ok",
  "data": { ... }
}
```

Error responses:

```json
{
  "detail": "Descriptive error message"
}
```

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- validation error |
| 404 | Not Found -- dataset, SLA, breach, or alert not found |
| 503 | Service Unavailable -- freshness monitor service not configured |
