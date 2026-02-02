# GL-019 HEATSCHEDULER REST API

## Overview

The GL-019 HEATSCHEDULER API provides programmatic access to the ProcessHeatingScheduler system, enabling automated scheduling of process heating operations to minimize energy costs while meeting production requirements.

**Agent ID:** GL-019
**Codename:** HEATSCHEDULER
**Version:** 1.0.0
**Base URL:** `https://api.greenlang.io/gl019`

---

## Table of Contents

1. [Authentication](#authentication)
2. [Rate Limiting](#rate-limiting)
3. [Error Handling](#error-handling)
4. [API Endpoints](#api-endpoints)
5. [Request/Response Examples](#requestresponse-examples)
6. [SDKs and Client Libraries](#sdks-and-client-libraries)
7. [Webhooks](#webhooks)
8. [Changelog](#changelog)

---

## Authentication

The API supports two authentication methods:

### OAuth2 Bearer Token

For user-based access and interactive applications.

```http
Authorization: Bearer <access_token>
```

**Obtaining a Token:**

```http
POST /api/v1/auth/token
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials&client_id=<client_id>&client_secret=<client_secret>
```

**Response:**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "scope": "schedules:read schedules:write equipment:read"
}
```

### API Key

For system-to-system integration and automation.

```http
X-API-Key: gl019_live_abc123xyz789
```

**API Key Prefixes:**

| Prefix | Environment |
|--------|-------------|
| `gl019_live_` | Production |
| `gl019_test_` | Staging/Test |
| `gl019_dev_` | Development |

### Scopes

| Scope | Description |
|-------|-------------|
| `schedules:read` | Read schedule data |
| `schedules:write` | Create and modify schedules |
| `equipment:read` | Read equipment data |
| `equipment:write` | Control equipment status |
| `analytics:read` | Access analytics and reports |
| `tariffs:read` | Read tariff data |
| `tariffs:write` | Upload custom tariffs |
| `demand-response:read` | Read DR status |
| `demand-response:write` | Respond to DR events |

---

## Rate Limiting

API requests are rate-limited based on your subscription tier:

| Tier | Requests/Minute | Requests/Hour | Burst |
|------|-----------------|---------------|-------|
| Free | 20 | 500 | 5 |
| Standard | 100 | 5,000 | 20 |
| Premium | 1,000 | 50,000 | 100 |
| Enterprise | Custom | Custom | Custom |

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1699531200
```

### Rate Limit Exceeded Response

```json
{
  "error": "rate_limit_exceeded",
  "message": "Rate limit exceeded. Please wait before retrying.",
  "retry_after": 60
}
```

---

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Resource created |
| 202 | Request accepted (async processing) |
| 204 | Success (no content) |
| 400 | Bad request - Invalid parameters |
| 401 | Unauthorized - Invalid or missing credentials |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not found |
| 409 | Conflict - Resource state conflict |
| 422 | Validation error |
| 429 | Rate limit exceeded |
| 500 | Internal server error |
| 503 | Service unavailable |

### Error Response Format

```json
{
  "error": "validation_error",
  "message": "Invalid request parameters",
  "details": [
    {
      "field": "start_date",
      "message": "start_date must be before end_date",
      "code": "invalid_date_range"
    }
  ],
  "request_id": "req_abc123xyz",
  "timestamp": "2025-11-09T10:30:00Z"
}
```

### Error Codes

| Code | Description |
|------|-------------|
| `validation_error` | Request validation failed |
| `authentication_error` | Authentication failed |
| `authorization_error` | Insufficient permissions |
| `not_found` | Resource not found |
| `conflict` | Resource state conflict |
| `rate_limit_exceeded` | Too many requests |
| `internal_error` | Internal server error |
| `service_unavailable` | Service temporarily unavailable |
| `schedule_locked` | Schedule cannot be modified |
| `equipment_unavailable` | Equipment not available |
| `tariff_expired` | Tariff data has expired |
| `erp_sync_failed` | ERP synchronization failed |

---

## API Endpoints

### Health Check

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health/liveness` | Kubernetes liveness probe |
| GET | `/health/readiness` | Kubernetes readiness probe |
| GET | `/metrics` | Prometheus metrics |

### Schedule Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/schedules/optimize` | Create optimized heating schedule |
| GET | `/api/v1/schedules/{schedule_id}` | Get schedule details |
| GET | `/api/v1/schedules` | List schedules with filtering |
| PUT | `/api/v1/schedules/{schedule_id}` | Update schedule |
| DELETE | `/api/v1/schedules/{schedule_id}` | Cancel schedule |

### Production Integration

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/production/batches` | Get production batches |
| POST | `/api/v1/production/sync` | Sync production schedule from ERP |

### Energy Tariffs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/tariffs/current` | Get current tariff rates |
| GET | `/api/v1/tariffs/forecast` | Get tariff forecast |
| POST | `/api/v1/tariffs` | Upload custom tariff |

### Equipment

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/equipment` | List heating equipment |
| GET | `/api/v1/equipment/{equipment_id}/availability` | Get equipment availability |
| PUT | `/api/v1/equipment/{equipment_id}/status` | Update equipment status |

### Analytics

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/analytics/savings` | Get savings report |
| GET | `/api/v1/analytics/forecast` | Get cost forecast |
| POST | `/api/v1/analytics/what-if` | Run what-if scenario |

### Demand Response

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/demand-response/event` | Handle DR event |
| GET | `/api/v1/demand-response/status` | Get DR status |

---

## Request/Response Examples

### Create Optimized Schedule

**Request:**

```http
POST /api/v1/schedules/optimize
Content-Type: application/json
Authorization: Bearer <token>

{
  "name": "Weekly Production Schedule",
  "description": "Optimized heating schedule for Week 45",
  "start_date": "2025-11-09",
  "end_date": "2025-11-15",
  "facility_id": "facility_01",
  "batch_ids": ["batch_001", "batch_002", "batch_003"],
  "objective": "minimize_cost",
  "constraints": {
    "avoid_peak_hours": true,
    "min_equipment_utilization": 0.7
  },
  "demand_response_enabled": true,
  "max_peak_demand_kw": 500.0
}
```

**Response (201 Created):**

```json
{
  "schedule_id": "sched_abc123xyz",
  "name": "Weekly Production Schedule",
  "description": "Optimized heating schedule for Week 45",
  "status": "draft",
  "start_date": "2025-11-09",
  "end_date": "2025-11-15",
  "facility_id": "facility_01",
  "objective": "minimize_cost",
  "operations": [
    {
      "operation_id": "op_001",
      "equipment_id": "furnace_01",
      "batch_id": "batch_001",
      "temperature_profile": {
        "initial_temp": 25.0,
        "target_temp": 850.0,
        "ramp_rate": 5.0,
        "hold_duration_minutes": 120,
        "tolerance": 2.0
      },
      "start_time": "2025-11-09T02:00:00Z",
      "end_time": "2025-11-09T05:30:00Z",
      "estimated_energy_kwh": 450.5,
      "estimated_cost": 36.04,
      "priority": "normal"
    }
  ],
  "total_energy_kwh": 1350.0,
  "total_cost": 108.12,
  "baseline_cost": 202.50,
  "savings": 94.38,
  "savings_percent": 46.6,
  "peak_demand_kw": 250.0,
  "created_at": "2025-11-09T10:00:00Z",
  "updated_at": "2025-11-09T10:00:00Z"
}
```

### Get Current Tariff

**Request:**

```http
GET /api/v1/tariffs/current?facility_id=facility_01
Authorization: Bearer <token>
```

**Response (200 OK):**

```json
{
  "tariff_id": "tariff_tou_2024",
  "name": "Time-of-Use Industrial Rate",
  "tariff_type": "time_of_use",
  "utility_provider": "Pacific Gas & Electric",
  "currency": "USD",
  "effective_date": "2024-01-01",
  "expiration_date": "2024-12-31",
  "periods": [
    {
      "start_time": "00:00:00",
      "end_time": "06:00:00",
      "rate_per_kwh": 0.08,
      "demand_charge_per_kw": 5.00,
      "period_name": "Off-Peak"
    },
    {
      "start_time": "06:00:00",
      "end_time": "14:00:00",
      "rate_per_kwh": 0.12,
      "demand_charge_per_kw": 10.00,
      "period_name": "Mid-Peak"
    },
    {
      "start_time": "14:00:00",
      "end_time": "20:00:00",
      "rate_per_kwh": 0.25,
      "demand_charge_per_kw": 20.00,
      "period_name": "On-Peak"
    },
    {
      "start_time": "20:00:00",
      "end_time": "00:00:00",
      "rate_per_kwh": 0.12,
      "demand_charge_per_kw": 10.00,
      "period_name": "Mid-Peak"
    }
  ],
  "demand_charge_per_kw": 15.00
}
```

### Handle Demand Response Event

**Request:**

```http
POST /api/v1/demand-response/event
Content-Type: application/json
Authorization: Bearer <token>

{
  "event_id": "dr_evt_20231109_001",
  "event_type": "curtailment",
  "facility_id": "facility_01",
  "start_time": "2025-11-09T14:00:00Z",
  "end_time": "2025-11-09T18:00:00Z",
  "required_reduction_kw": 200.0,
  "incentive_rate": 0.50,
  "penalty_rate": 1.00,
  "mandatory": false,
  "notification_lead_time_minutes": 60
}
```

**Response (202 Accepted):**

```json
{
  "event_id": "dr_evt_20231109_001",
  "facility_id": "facility_01",
  "participation_status": "accepted",
  "committed_reduction_kw": 150.0,
  "estimated_revenue": 300.00,
  "rescheduled_operations": 3,
  "response_received_at": "2025-11-09T12:00:00Z"
}
```

### Get Savings Report

**Request:**

```http
GET /api/v1/analytics/savings?facility_id=facility_01&start_date=2025-11-01&end_date=2025-11-30&include_breakdown=true
Authorization: Bearer <token>
```

**Response (200 OK):**

```json
{
  "report_id": "rpt_sav_202311",
  "facility_id": "facility_01",
  "period_start": "2025-11-01",
  "period_end": "2025-11-30",
  "total_energy_kwh": 150000.0,
  "total_cost": 12000.00,
  "baseline_cost": 15000.00,
  "total_savings": 3000.00,
  "savings_percent": 20.0,
  "breakdown": {
    "time_shifting": 1800.00,
    "demand_reduction": 600.00,
    "efficiency_improvement": 300.00,
    "demand_response": 300.00
  },
  "schedules_optimized": 45,
  "co2_avoided_kg": 1500.0,
  "generated_at": "2025-12-01T08:00:00Z"
}
```

---

## SDKs and Client Libraries

### Python SDK

```bash
pip install greenlang-heatscheduler
```

```python
from greenlang import HeatSchedulerClient

client = HeatSchedulerClient(
    api_key="gl019_live_abc123xyz789",
    base_url="https://api.greenlang.io/gl019"
)

# Create optimized schedule
schedule = client.schedules.optimize(
    name="Weekly Production Schedule",
    start_date="2025-11-09",
    end_date="2025-11-15",
    facility_id="facility_01",
    batch_ids=["batch_001", "batch_002"],
    objective="minimize_cost"
)

print(f"Schedule created: {schedule.schedule_id}")
print(f"Estimated savings: ${schedule.savings:.2f} ({schedule.savings_percent:.1f}%)")
```

### JavaScript/TypeScript SDK

```bash
npm install @greenlang/heatscheduler
```

```typescript
import { HeatSchedulerClient } from '@greenlang/heatscheduler';

const client = new HeatSchedulerClient({
  apiKey: 'gl019_live_abc123xyz789',
  baseUrl: 'https://api.greenlang.io/gl019'
});

// Get current tariff
const tariff = await client.tariffs.getCurrent({
  facilityId: 'facility_01'
});

console.log(`Current tariff: ${tariff.name}`);
console.log(`Peak rate: $${tariff.periods.find(p => p.periodName === 'On-Peak')?.ratePerKwh}/kWh`);
```

### cURL Examples

```bash
# Get schedule details
curl -X GET "https://api.greenlang.io/gl019/api/v1/schedules/sched_abc123" \
  -H "Authorization: Bearer <token>"

# Update equipment status
curl -X PUT "https://api.greenlang.io/gl019/api/v1/equipment/furnace_01/status" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"status": "maintenance", "reason": "Scheduled preventive maintenance", "expected_duration_hours": 8.0}'
```

---

## Webhooks

Subscribe to real-time events via webhooks.

### Available Events

| Event | Description |
|-------|-------------|
| `schedule.created` | New schedule created |
| `schedule.approved` | Schedule approved |
| `schedule.started` | Schedule execution started |
| `schedule.completed` | Schedule execution completed |
| `schedule.failed` | Schedule execution failed |
| `equipment.status_changed` | Equipment status changed |
| `demand_response.event_received` | DR event received |
| `demand_response.response_sent` | DR response submitted |

### Webhook Payload

```json
{
  "id": "evt_abc123xyz",
  "type": "schedule.completed",
  "created_at": "2025-11-09T18:00:00Z",
  "data": {
    "schedule_id": "sched_abc123",
    "facility_id": "facility_01",
    "actual_energy_kwh": 14500.0,
    "actual_cost": 1160.00,
    "actual_savings": 340.00
  }
}
```

### Webhook Security

All webhook payloads are signed with HMAC-SHA256:

```http
X-GreenLang-Signature: sha256=abc123...
X-GreenLang-Timestamp: 1699531200
```

Verify the signature:

```python
import hmac
import hashlib

def verify_webhook(payload: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)
```

---

## Changelog

### v1.0.0 (2025-11-09)

- Initial release
- Schedule optimization endpoints
- Production batch integration
- Energy tariff management
- Equipment monitoring and control
- Cost analytics and forecasting
- Demand response support

---

## Support

- **Documentation:** https://docs.greenlang.io/gl019
- **API Status:** https://status.greenlang.io
- **Support Email:** api-support@greenlang.io
- **Community Forum:** https://community.greenlang.io

---

*Copyright 2025 GreenLang. All rights reserved.*
