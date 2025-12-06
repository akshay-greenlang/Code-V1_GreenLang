# GreenLang Process Heat API Reference

## Overview

The GreenLang Process Heat API provides programmatic access to industrial process heat optimization, emissions monitoring, and regulatory compliance capabilities. This API enables integration with industrial control systems, enterprise software, and third-party applications.

**Current Version:** v1.0.0
**API Base URL:** `https://api.greenlang.io/v1/process-heat`
**Documentation Updated:** December 2025

---

## Table of Contents

1. [Authentication](#authentication)
2. [Base URL and Versioning](#base-url-and-versioning)
3. [Rate Limiting and Quotas](#rate-limiting-and-quotas)
4. [Request and Response Formats](#request-and-response-formats)
5. [Error Handling](#error-handling)
6. [API Endpoints Overview](#api-endpoints-overview)
7. [Webhooks](#webhooks)
8. [SDK Libraries](#sdk-libraries)

---

## Authentication

The Process Heat API uses OAuth 2.0 with JWT (JSON Web Tokens) for authentication. All API requests must include a valid access token in the Authorization header.

### Obtaining an Access Token

```http
POST /api/v1/auth/token
Content-Type: application/x-www-form-urlencoded
```

**Request Body:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `grant_type` | string | Yes | Must be `client_credentials` |
| `client_id` | string | Yes | Your application's client ID |
| `client_secret` | string | Yes | Your application's client secret |
| `scope` | string | No | Space-separated list of scopes (default: `read write`) |

**Example Request:**

```bash
curl -X POST "https://api.greenlang.io/v1/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials" \
  -d "client_id=your_client_id" \
  -d "client_secret=your_client_secret" \
  -d "scope=read write"
```

**Response (200 OK):**

```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "scope": "read write",
  "issued_at": "2025-12-06T10:30:00Z"
}
```

### Using the Access Token

Include the access token in the `Authorization` header for all API requests:

```http
Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Example:**

```bash
curl -X GET "https://api.greenlang.io/v1/process-heat/status" \
  -H "Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### API Key Authentication (Alternative)

For server-to-server integrations, you can use API key authentication:

```http
X-API-Key: your_api_key_here
```

### Scopes

| Scope | Description |
|-------|-------------|
| `read` | Read access to resources |
| `write` | Write access to resources |
| `admin` | Administrative operations |
| `emissions:read` | Read emissions data |
| `emissions:write` | Submit emissions data |
| `compliance:read` | Read compliance reports |
| `compliance:write` | Generate compliance reports |
| `safety:read` | Read safety status |
| `safety:write` | Manage safety permits and ESD |

---

## Base URL and Versioning

### Base URL

All API endpoints are relative to the base URL:

| Environment | Base URL |
|-------------|----------|
| Production | `https://api.greenlang.io/v1` |
| Staging | `https://staging-api.greenlang.io/v1` |
| Development | `https://dev-api.greenlang.io/v1` |

### API Versioning

The API version is included in the URL path. The current version is `v1`.

```
https://api.greenlang.io/v1/process-heat/calculate
                        ^^
                        API version
```

**Version Lifecycle:**

| Version | Status | Sunset Date |
|---------|--------|-------------|
| v1 | Current | - |
| v0 (Beta) | Deprecated | 2025-06-01 |

### Request Headers

All requests should include the following headers:

| Header | Required | Description |
|--------|----------|-------------|
| `Authorization` | Yes | Bearer token or API key |
| `Content-Type` | Yes (for POST/PUT) | `application/json` |
| `Accept` | Recommended | `application/json` |
| `X-Request-ID` | Recommended | Unique request identifier for tracing |
| `X-API-Version` | Optional | API version override (e.g., `2025-12-01`) |

---

## Rate Limiting and Quotas

### Rate Limits

The API enforces rate limits to ensure fair usage and system stability:

| Tier | Requests/Minute | Requests/Hour | Requests/Day |
|------|-----------------|---------------|--------------|
| Free | 60 | 1,000 | 10,000 |
| Professional | 300 | 10,000 | 100,000 |
| Enterprise | 1,000 | 50,000 | 500,000 |
| Unlimited | Custom | Custom | Custom |

### Rate Limit Headers

Rate limit information is included in all API responses:

| Header | Description |
|--------|-------------|
| `X-RateLimit-Limit` | Maximum requests allowed in the current window |
| `X-RateLimit-Remaining` | Remaining requests in the current window |
| `X-RateLimit-Reset` | Unix timestamp when the rate limit resets |
| `Retry-After` | Seconds to wait before retrying (when rate limited) |

**Example Response Headers:**

```http
X-RateLimit-Limit: 300
X-RateLimit-Remaining: 245
X-RateLimit-Reset: 1733486400
```

### Rate Limit Exceeded Response

When rate limit is exceeded, the API returns a `429 Too Many Requests` response:

```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again in 42 seconds.",
    "retry_after": 42
  },
  "timestamp": "2025-12-06T10:30:00Z"
}
```

### Best Practices

1. **Implement exponential backoff** when receiving 429 responses
2. **Use the `Retry-After` header** to determine when to retry
3. **Batch requests** when possible to stay within limits
4. **Cache responses** to reduce API calls
5. **Use webhooks** for real-time updates instead of polling

---

## Request and Response Formats

### Request Format

All request bodies must be valid JSON with UTF-8 encoding:

```http
POST /api/v1/process-heat/calculate
Content-Type: application/json
Authorization: Bearer {access_token}

{
  "workflow_type": "optimization",
  "parameters": {
    "boiler_id": "BLR-001",
    "target_efficiency": 92.5
  }
}
```

### Response Format

All responses follow a consistent envelope structure:

**Success Response:**

```json
{
  "success": true,
  "data": {
    "job_id": "job_abc123",
    "status": "completed",
    "results": { ... }
  },
  "metadata": {
    "request_id": "req_xyz789",
    "processing_time_ms": 245,
    "provenance_hash": "sha256:a1b2c3d4..."
  },
  "timestamp": "2025-12-06T10:30:00Z"
}
```

**Error Response:**

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid boiler_id format",
    "details": {
      "field": "boiler_id",
      "constraint": "Must match pattern BLR-[0-9]{3}"
    }
  },
  "metadata": {
    "request_id": "req_xyz789"
  },
  "timestamp": "2025-12-06T10:30:00Z"
}
```

### Pagination

For endpoints returning lists, pagination is supported:

**Request:**

```http
GET /api/v1/process-heat/jobs?page=1&per_page=50&sort=created_at&order=desc
```

**Pagination Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | integer | 1 | Page number (1-indexed) |
| `per_page` | integer | 20 | Items per page (max: 100) |
| `sort` | string | `created_at` | Field to sort by |
| `order` | string | `desc` | Sort order (`asc` or `desc`) |

**Paginated Response:**

```json
{
  "success": true,
  "data": {
    "items": [...],
    "pagination": {
      "page": 1,
      "per_page": 50,
      "total_items": 1250,
      "total_pages": 25,
      "has_next": true,
      "has_prev": false
    }
  },
  "timestamp": "2025-12-06T10:30:00Z"
}
```

### Date and Time Formats

All dates and times use ISO 8601 format in UTC:

```
2025-12-06T10:30:00Z
```

For duration fields, use ISO 8601 duration format:

```
PT1H30M (1 hour 30 minutes)
P7D (7 days)
```

---

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| `200` | Success |
| `201` | Created |
| `202` | Accepted (async processing started) |
| `204` | No Content |
| `400` | Bad Request - Invalid input |
| `401` | Unauthorized - Invalid or missing authentication |
| `403` | Forbidden - Insufficient permissions |
| `404` | Not Found - Resource does not exist |
| `409` | Conflict - Resource already exists |
| `422` | Unprocessable Entity - Validation failed |
| `429` | Too Many Requests - Rate limit exceeded |
| `500` | Internal Server Error |
| `502` | Bad Gateway |
| `503` | Service Unavailable |
| `504` | Gateway Timeout |

### Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `AUTH_INVALID_TOKEN` | Token is invalid or expired | Refresh your access token |
| `AUTH_INSUFFICIENT_SCOPE` | Token lacks required scope | Request token with correct scopes |
| `VALIDATION_ERROR` | Request validation failed | Check the `details` field for specifics |
| `RESOURCE_NOT_FOUND` | Requested resource not found | Verify the resource ID |
| `RESOURCE_CONFLICT` | Resource already exists | Use a different identifier |
| `RATE_LIMIT_EXCEEDED` | Rate limit exceeded | Wait and retry with backoff |
| `QUOTA_EXCEEDED` | Monthly quota exceeded | Upgrade your plan or wait |
| `CALCULATION_ERROR` | Calculation failed | Check input parameters |
| `SAFETY_VIOLATION` | Safety constraint violated | Review safety parameters |
| `EQUIPMENT_OFFLINE` | Target equipment is offline | Check equipment connectivity |
| `WORKFLOW_TIMEOUT` | Workflow execution timed out | Increase timeout or simplify workflow |
| `INTERNAL_ERROR` | Internal server error | Contact support with request_id |

### Error Response Structure

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Human-readable error message",
    "details": {
      "field": "steam_pressure_psig",
      "constraint": "Must be between 0 and 3000",
      "received": -50
    },
    "documentation_url": "https://docs.greenlang.io/errors/VALIDATION_ERROR"
  },
  "metadata": {
    "request_id": "req_xyz789"
  },
  "timestamp": "2025-12-06T10:30:00Z"
}
```

---

## API Endpoints Overview

### Process Heat Orchestrator (GL-001)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/process-heat/calculate` | Submit calculation workflow |
| `GET` | `/process-heat/status/{job_id}` | Get job status |
| `GET` | `/process-heat/results/{job_id}` | Get job results |
| `DELETE` | `/process-heat/jobs/{job_id}` | Cancel a running job |

[View Orchestrator API Documentation](endpoints/orchestrator.md)

### Emissions APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/emissions/carbon` | Calculate carbon emissions |
| `POST` | `/emissions/scope3` | Calculate Scope 3 emissions |
| `POST` | `/emissions/batch` | Batch emissions calculation |
| `GET` | `/emissions/factors` | Get emission factors |
| `GET` | `/emissions/sources/{source_id}` | Get source emissions |

[View Emissions API Documentation](endpoints/emissions.md)

### Compliance APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/compliance/csrd` | Generate CSRD report |
| `POST` | `/compliance/cbam` | Calculate CBAM obligations |
| `POST` | `/compliance/eudr` | EUDR compliance check |
| `GET` | `/compliance/deadlines` | Get regulatory deadlines |
| `GET` | `/compliance/reports/{report_id}` | Get compliance report |

[View Compliance API Documentation](endpoints/compliance.md)

### Agent Management APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/agents` | List registered agents |
| `POST` | `/agents` | Register a new agent |
| `GET` | `/agents/{agent_id}` | Get agent status |
| `DELETE` | `/agents/{agent_id}` | Deregister an agent |
| `GET` | `/agents/{agent_id}/metrics` | Get agent metrics |

### Safety APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/safety/status` | Get safety system status |
| `POST` | `/safety/esd` | Trigger emergency shutdown |
| `DELETE` | `/safety/esd` | Reset emergency shutdown |
| `GET` | `/safety/permits` | List safety permits |
| `POST` | `/safety/permits` | Request safety permit |

---

## Webhooks

Configure webhooks to receive real-time notifications about events.

### Configuring Webhooks

```http
POST /api/v1/webhooks
Content-Type: application/json
Authorization: Bearer {access_token}

{
  "url": "https://your-app.com/webhooks/greenlang",
  "events": [
    "job.completed",
    "job.failed",
    "emission.exceedance",
    "safety.alarm"
  ],
  "secret": "your_webhook_secret",
  "active": true
}
```

### Webhook Events

| Event | Description |
|-------|-------------|
| `job.created` | Calculation job created |
| `job.started` | Job execution started |
| `job.completed` | Job completed successfully |
| `job.failed` | Job failed |
| `emission.exceedance` | Emission limit exceeded |
| `emission.warning` | Emission approaching limit |
| `safety.alarm` | Safety alarm triggered |
| `safety.esd` | Emergency shutdown triggered |
| `compliance.deadline` | Compliance deadline approaching |
| `compliance.report_ready` | Compliance report generated |

### Webhook Payload

```json
{
  "event": "job.completed",
  "timestamp": "2025-12-06T10:35:00Z",
  "data": {
    "job_id": "job_abc123",
    "status": "completed",
    "results_url": "https://api.greenlang.io/v1/process-heat/results/job_abc123"
  },
  "signature": "sha256=a1b2c3d4e5f6..."
}
```

### Verifying Webhook Signatures

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

## SDK Libraries

Official SDK libraries are available for common programming languages:

### Python

```bash
pip install greenlang-sdk
```

```python
from greenlang import ProcessHeatClient

client = ProcessHeatClient(
    client_id="your_client_id",
    client_secret="your_client_secret"
)

# Submit calculation
job = client.calculate(
    workflow_type="optimization",
    parameters={
        "boiler_id": "BLR-001",
        "target_efficiency": 92.5
    }
)

# Wait for completion
result = client.wait_for_job(job.job_id)

# Get emissions
emissions = client.emissions.calculate_carbon(
    fuel_type="natural_gas",
    fuel_consumption=100.0,
    fuel_unit="MMBTU"
)
```

### JavaScript/TypeScript

```bash
npm install @greenlang/process-heat-sdk
```

```typescript
import { ProcessHeatClient } from '@greenlang/process-heat-sdk';

const client = new ProcessHeatClient({
  clientId: 'your_client_id',
  clientSecret: 'your_client_secret'
});

// Submit calculation
const job = await client.calculate({
  workflowType: 'optimization',
  parameters: {
    boilerId: 'BLR-001',
    targetEfficiency: 92.5
  }
});

// Get results
const result = await client.getResults(job.jobId);
```

### Go

```bash
go get github.com/greenlang/process-heat-sdk-go
```

```go
import "github.com/greenlang/process-heat-sdk-go"

client := processheat.NewClient(
    processheat.WithCredentials(clientID, clientSecret),
)

job, err := client.Calculate(ctx, &processheat.CalculateRequest{
    WorkflowType: "optimization",
    Parameters: map[string]interface{}{
        "boiler_id": "BLR-001",
        "target_efficiency": 92.5,
    },
})
```

---

## Support

- **Documentation:** https://docs.greenlang.io/process-heat
- **API Status:** https://status.greenlang.io
- **Support Email:** support@greenlang.io
- **Community Forum:** https://community.greenlang.io
- **GitHub Issues:** https://github.com/greenlang/process-heat-api/issues

---

## Changelog

### v1.0.0 (2025-12-01)
- Initial production release
- Complete orchestrator API
- Emissions calculation endpoints
- Compliance reporting (CSRD, CBAM, EUDR)
- Safety management APIs
- Webhook support

### v0.9.0 (2025-10-01) - Beta
- Beta release for early adopters
- Core calculation endpoints
- Basic emissions tracking
