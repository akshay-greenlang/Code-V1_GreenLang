# Audit Logging Service API Reference (SEC-005)

## Overview

The Centralized Audit Logging Service provides comprehensive audit event querying, advanced search, statistics and analytics, real-time event streaming via WebSocket, compliance report generation (SOC 2, ISO 27001, GDPR), and data export capabilities. Backed by TimescaleDB with continuous aggregates for efficient analytics.

**Router Prefix:** `/api/v1/audit`
**Tags:** `Audit Events`, `Audit Search`, `Audit Statistics`, `Audit Streaming`, `Audit Reports`, `Audit Export`
**Source:** `greenlang/infrastructure/audit_service/api/`

---

## Endpoint Summary

### Events

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/api/v1/audit/events` | List events with filters and pagination | Yes |
| GET | `/api/v1/audit/events/{event_id}` | Get single event details | Yes |

### Search

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| POST | `/api/v1/audit/search` | Advanced search with aggregations | Yes |

### Statistics

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/api/v1/audit/stats` | Event statistics (counts by type, severity) | Yes |
| GET | `/api/v1/audit/timeline` | Activity timeline for user/resource | Yes |
| GET | `/api/v1/audit/hotspots` | Top 10 users, resources, IPs | Yes |

### Streaming

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| WebSocket | `/api/v1/audit/events/stream` | Real-time event subscription | Yes (JWT) |

### Compliance Reports

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| POST | `/api/v1/audit/reports/soc2` | Generate SOC 2 compliance report | Yes |
| POST | `/api/v1/audit/reports/iso27001` | Generate ISO 27001 report | Yes |
| POST | `/api/v1/audit/reports/gdpr` | Generate GDPR report | Yes |
| GET | `/api/v1/audit/reports/{job_id}` | Get report generation status | Yes |
| GET | `/api/v1/audit/reports/{job_id}/download` | Download generated report | Yes |

### Export

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| POST | `/api/v1/audit/export` | Start export job (CSV, JSON, Parquet) | Yes |
| GET | `/api/v1/audit/export/{job_id}` | Get export job status | Yes |
| GET | `/api/v1/audit/export/{job_id}/download` | Download export | Yes |

---

## Event Endpoints

### GET /api/v1/audit/events

List audit events with filters and pagination. Results are ordered newest-first.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `category` | string | - | Filter by event category |
| `severity` | string | - | Filter by severity level |
| `event_type` | string | - | Filter by event type identifier |
| `user_id` | UUID | - | Filter by acting user |
| `tenant_id` | UUID | - | Filter by tenant |
| `since` | datetime | - | Events after this timestamp (ISO 8601) |
| `until` | datetime | - | Events before this timestamp (ISO 8601) |
| `page` | integer | 1 | Page number |
| `page_size` | integer | 50 | Items per page (1-200) |

**Response (200 OK):**

```json
{
  "events": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "event_id": "corr-abc123",
      "trace_id": "trace-xyz789",
      "performed_at": "2026-04-04T10:30:00Z",
      "category": "authentication",
      "severity": "info",
      "event_type": "login_success",
      "operation": "login",
      "user_id": "550e8400-e29b-41d4-a716-446655440001"
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 50,
  "has_next": false
}
```

---

## Search Endpoint

### POST /api/v1/audit/search

Advanced search with aggregations. Supports LogQL-like query syntax.

**Query Syntax:**

- `field:value` -- exact match
- `field:"quoted value"` -- exact match with spaces
- `field:value*` -- prefix match
- `AND`, `OR` operators

**Request Body:**

```json
{
  "query": "category:authentication AND severity:warning",
  "time_range": {
    "start": "2026-04-01T00:00:00Z",
    "end": "2026-04-04T23:59:59Z"
  },
  "page": 1,
  "page_size": 50,
  "aggregations": ["category", "severity"]
}
```

---

## Statistics Endpoints

### GET /api/v1/audit/stats

Event statistics using TimescaleDB continuous aggregates. Returns counts by category, severity, and operation for a time range.

### GET /api/v1/audit/timeline

Activity timeline for a specific user or resource, showing event frequency over time.

### GET /api/v1/audit/hotspots

Top 10 most active users, resources, and IP addresses based on event volume.

---

## WebSocket Streaming

### WebSocket /api/v1/audit/events/stream

Real-time audit event subscription via WebSocket. Subscribes to the `gl:audit:events` Redis pub/sub channel.

**Connection:**

```
ws://api.greenlang.io/api/v1/audit/events/stream?token={jwt}&types=login_success,login_failure&tenant_id=t-acme-corp
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `token` | string | JWT access token for authentication |
| `types` | string | Comma-separated event types to filter |
| `tenant_id` | string | Tenant filter |
| `severity` | string | Minimum severity filter |

**Features:**

- 30-second heartbeat for connection health
- Event filtering by type, tenant, and severity
- JWT authentication required
- Automatic reconnection support (max 3 attempts)

---

## Compliance Reports

### POST /api/v1/audit/reports/soc2

Generate a SOC 2 compliance report for a specified period. Returns a job ID for async generation.

**Request Body:**

```json
{
  "period": "last_30_days",
  "format": "pdf",
  "tenant_id": "t-acme-corp",
  "include_details": true
}
```

**Response (202 Accepted):**

```json
{
  "job_id": "rpt-abc123",
  "report_type": "soc2",
  "status": "generating",
  "estimated_completion": "2026-04-04T11:15:00Z"
}
```

The same pattern applies to ISO 27001 and GDPR report endpoints.

---

## Export

### POST /api/v1/audit/export

Start an export job. Supports CSV, JSON, and Parquet formats.

**Request Body:**

```json
{
  "filters": {
    "since": "2026-01-01T00:00:00Z",
    "until": "2026-03-31T23:59:59Z",
    "event_types": ["login_success", "login_failure"],
    "tenant_id": "t-acme-corp"
  },
  "format": "csv",
  "compression": "gzip"
}
```

**Response (202 Accepted):**

```json
{
  "job_id": "exp-def456",
  "status": "processing",
  "format": "csv",
  "estimated_size_mb": 12.5
}
```
