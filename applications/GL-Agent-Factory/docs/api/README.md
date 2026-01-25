# GreenLang API Reference

## Overview

The GreenLang API provides programmatic access to carbon accounting, emissions calculations, and regulatory compliance tools. Build applications that calculate greenhouse gas emissions, generate compliance reports, and integrate sustainability data into your existing workflows.

**Base URL:** `https://api.greenlang.io/v1`

**API Version:** v1 (current), v0 (deprecated)

---

## Authentication

GreenLang supports two authentication methods:

### 1. JWT Bearer Token (Recommended)

Obtain an access token using OAuth 2.0 client credentials flow:

```http
POST /api/v1/auth/token
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials&
client_id=your_client_id&
client_secret=your_client_secret
```

**Response:**

```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "scope": "read write calculate"
}
```

**Usage:**

Include the token in the `Authorization` header:

```http
Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Python Example:**

```python
import requests

# Obtain access token
response = requests.post(
    "https://api.greenlang.io/v1/auth/token",
    data={
        "grant_type": "client_credentials",
        "client_id": "your_client_id",
        "client_secret": "your_client_secret"
    }
)

token_data = response.json()
access_token = token_data["access_token"]

# Use token for API requests
headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}
```

**cURL Example:**

```bash
# Obtain token
TOKEN=$(curl -s -X POST "https://api.greenlang.io/v1/auth/token" \
  -d "grant_type=client_credentials" \
  -d "client_id=your_client_id" \
  -d "client_secret=your_client_secret" | jq -r '.access_token')

# Use token
curl -H "Authorization: Bearer $TOKEN" \
  "https://api.greenlang.io/v1/agents"
```

**JavaScript Example:**

```javascript
const axios = require('axios');

// Obtain access token
const tokenResponse = await axios.post(
  'https://api.greenlang.io/v1/auth/token',
  new URLSearchParams({
    grant_type: 'client_credentials',
    client_id: 'your_client_id',
    client_secret: 'your_client_secret'
  })
);

const accessToken = tokenResponse.data.access_token;

// Create authenticated client
const apiClient = axios.create({
  baseURL: 'https://api.greenlang.io/v1',
  headers: {
    'Authorization': `Bearer ${accessToken}`,
    'Content-Type': 'application/json'
  }
});
```

### 2. API Key

For server-to-server integrations, use an API key:

```http
X-API-Key: gl_live_abc123xyz789
```

**API Key Types:**

| Type | Prefix | Usage |
|------|--------|-------|
| Live | `gl_live_` | Production environment |
| Test | `gl_test_` | Sandbox/testing environment |

**Important:** Never expose API keys in client-side code or version control.

---

## Base URL and Versioning

### Base URLs

| Environment | URL |
|-------------|-----|
| Production | `https://api.greenlang.io/v1` |
| Staging | `https://staging-api.greenlang.io/v1` |
| Sandbox | `https://sandbox-api.greenlang.io/v1` |

### API Versioning

The API version is included in the URL path. The current version is `v1`.

```
https://api.greenlang.io/v1/agents
https://api.greenlang.io/v1/calculate/fuel
```

**Version Lifecycle:**

| Status | Description |
|--------|-------------|
| Current | Fully supported, receives new features |
| Deprecated | Supported but no new features, migration recommended |
| Sunset | End of life, returns 410 Gone |

**Version Headers:**

```http
X-API-Version: 2025-01-15
X-Deprecation-Notice: This endpoint will be removed on 2026-01-01
```

---

## Rate Limits

Rate limits protect the API from abuse and ensure fair usage for all users.

### Limits by Plan

| Plan | Requests/Minute | Requests/Day | Concurrent |
|------|-----------------|--------------|------------|
| Free | 10 | 1,000 | 2 |
| Starter | 60 | 10,000 | 5 |
| Professional | 300 | 100,000 | 20 |
| Enterprise | 1,000 | Unlimited | 100 |

### Rate Limit Headers

Every response includes rate limit information:

```http
X-RateLimit-Limit: 300
X-RateLimit-Remaining: 298
X-RateLimit-Reset: 1699531200
X-RateLimit-Reset-After: 45
```

| Header | Description |
|--------|-------------|
| `X-RateLimit-Limit` | Maximum requests per window |
| `X-RateLimit-Remaining` | Requests remaining in current window |
| `X-RateLimit-Reset` | Unix timestamp when window resets |
| `X-RateLimit-Reset-After` | Seconds until window resets |

### Rate Limit Exceeded (429)

When rate limit is exceeded:

```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Rate limit exceeded. Please retry after 45 seconds.",
    "retry_after": 45
  }
}
```

**Best Practices:**

1. Implement exponential backoff with jitter
2. Cache responses when possible
3. Use webhooks instead of polling
4. Batch requests when available

**Python Retry Example:**

```python
import time
import random
import requests

def make_request_with_retry(url, headers, max_retries=5):
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)

        if response.status_code == 429:
            retry_after = int(response.headers.get('X-RateLimit-Reset-After', 60))
            jitter = random.uniform(0, 1)
            sleep_time = retry_after + jitter
            print(f"Rate limited. Retrying in {sleep_time:.1f}s...")
            time.sleep(sleep_time)
            continue

        return response

    raise Exception("Max retries exceeded")
```

---

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | OK - Request succeeded |
| 201 | Created - Resource created |
| 202 | Accepted - Request accepted for async processing |
| 204 | No Content - Success with no response body |
| 400 | Bad Request - Invalid request parameters |
| 401 | Unauthorized - Invalid or missing authentication |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource does not exist |
| 409 | Conflict - Resource already exists or state conflict |
| 422 | Unprocessable Entity - Validation error |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error - Server-side error |
| 502 | Bad Gateway - Upstream service error |
| 503 | Service Unavailable - Temporary outage |

### Error Response Format

All errors follow a consistent format:

```json
{
  "error": {
    "code": "validation_error",
    "message": "Request validation failed",
    "request_id": "req_abc123xyz789",
    "details": [
      {
        "field": "fuel_quantity",
        "message": "Must be a positive number",
        "code": "invalid_value"
      },
      {
        "field": "fuel_type",
        "message": "Unknown fuel type: 'petrol'. Did you mean 'gasoline'?",
        "code": "unknown_value",
        "suggestion": "gasoline"
      }
    ],
    "documentation_url": "https://docs.greenlang.io/errors/validation_error"
  }
}
```

### Error Codes

| Code | Description |
|------|-------------|
| `authentication_error` | Invalid credentials or token |
| `authorization_error` | Insufficient permissions |
| `validation_error` | Request body validation failed |
| `not_found` | Resource not found |
| `rate_limit_exceeded` | Too many requests |
| `calculation_error` | Emissions calculation failed |
| `data_quality_error` | Input data quality issues |
| `external_service_error` | Third-party service unavailable |
| `internal_error` | Unexpected server error |

### Error Handling Example

**Python:**

```python
import requests

def call_api(endpoint, data):
    try:
        response = requests.post(
            f"https://api.greenlang.io/v1/{endpoint}",
            json=data,
            headers={"Authorization": f"Bearer {token}"}
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as e:
        error = e.response.json().get("error", {})

        if error.get("code") == "validation_error":
            for detail in error.get("details", []):
                print(f"Field '{detail['field']}': {detail['message']}")

        elif error.get("code") == "rate_limit_exceeded":
            retry_after = error.get("retry_after", 60)
            print(f"Rate limited. Retry after {retry_after}s")

        else:
            print(f"API Error: {error.get('message')}")

        raise
```

**JavaScript:**

```javascript
async function callApi(endpoint, data) {
  try {
    const response = await fetch(`https://api.greenlang.io/v1/${endpoint}`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    });

    if (!response.ok) {
      const errorData = await response.json();
      const error = errorData.error;

      if (error.code === 'validation_error') {
        error.details.forEach(detail => {
          console.error(`Field '${detail.field}': ${detail.message}`);
        });
      }

      throw new Error(error.message);
    }

    return await response.json();
  } catch (error) {
    console.error('API call failed:', error.message);
    throw error;
  }
}
```

---

## Response Format

### Standard Response Structure

All successful responses follow this format:

```json
{
  "data": {
    // Response payload
  },
  "meta": {
    "request_id": "req_abc123xyz789",
    "timestamp": "2025-01-15T10:30:00Z",
    "version": "v1"
  }
}
```

### List Response with Pagination

```json
{
  "data": [
    { "id": "agent_001", "name": "Fuel Emissions Calculator" },
    { "id": "agent_002", "name": "CBAM Reporter" }
  ],
  "meta": {
    "request_id": "req_abc123xyz789",
    "timestamp": "2025-01-15T10:30:00Z"
  },
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total_items": 45,
    "total_pages": 3,
    "has_next": true,
    "has_prev": false
  }
}
```

### Pagination Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | integer | 1 | Page number (1-indexed) |
| `per_page` | integer | 20 | Items per page (max: 100) |
| `sort` | string | `created_at` | Sort field |
| `order` | string | `desc` | Sort order: `asc` or `desc` |

**Example:**

```http
GET /v1/agents?page=2&per_page=50&sort=name&order=asc
```

### Async Operation Response

For long-running operations:

```json
{
  "data": {
    "job_id": "job_abc123",
    "status": "processing",
    "progress": 45,
    "estimated_completion": "2025-01-15T10:35:00Z"
  },
  "meta": {
    "request_id": "req_xyz789",
    "status_url": "/v1/jobs/job_abc123"
  }
}
```

---

## Request Headers

### Required Headers

| Header | Description |
|--------|-------------|
| `Authorization` | Bearer token or API key |
| `Content-Type` | `application/json` for POST/PUT/PATCH |

### Optional Headers

| Header | Description |
|--------|-------------|
| `Accept` | Response format (default: `application/json`) |
| `Accept-Language` | Response language (default: `en`) |
| `X-Idempotency-Key` | Unique key for idempotent requests |
| `X-Request-ID` | Client-provided request ID for tracing |

### Idempotency

For POST requests that should be idempotent:

```http
POST /v1/calculate/fuel
X-Idempotency-Key: unique-request-id-12345
Content-Type: application/json

{...}
```

Replaying the same request with the same idempotency key returns the original response without reprocessing.

---

## Webhooks

Configure webhooks to receive real-time notifications:

```http
POST /v1/webhooks
Content-Type: application/json
Authorization: Bearer {access_token}

{
  "url": "https://your-app.com/webhooks/greenlang",
  "events": ["job.completed", "job.failed", "report.generated"],
  "secret": "your_webhook_secret"
}
```

### Webhook Events

| Event | Description |
|-------|-------------|
| `job.started` | Async job has started processing |
| `job.completed` | Async job completed successfully |
| `job.failed` | Async job failed |
| `report.generated` | Compliance report is ready |
| `alert.threshold` | Emissions threshold exceeded |

### Webhook Payload

```json
{
  "event": "job.completed",
  "timestamp": "2025-01-15T10:35:00Z",
  "data": {
    "job_id": "job_abc123",
    "status": "completed",
    "results_url": "/v1/jobs/job_abc123/results"
  }
}
```

### Webhook Signature Verification

Verify webhook authenticity using HMAC-SHA256:

```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    expected = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(f"sha256={expected}", signature)
```

---

## SDKs and Libraries

Official SDKs are available for popular languages:

| Language | Package | Installation |
|----------|---------|--------------|
| Python | `greenlang-sdk` | `pip install greenlang-sdk` |
| JavaScript | `@greenlang/sdk` | `npm install @greenlang/sdk` |
| Go | `github.com/greenlang/sdk-go` | `go get github.com/greenlang/sdk-go` |
| Java | `io.greenlang:sdk` | Maven/Gradle |
| .NET | `GreenLang.SDK` | `dotnet add package GreenLang.SDK` |

---

## Support and Resources

- **API Status:** https://status.greenlang.io
- **Documentation:** https://docs.greenlang.io
- **Community Forum:** https://community.greenlang.io
- **Support Email:** support@greenlang.io
- **GitHub:** https://github.com/greenlang

---

## Next Steps

- [Agents API](./agents.md) - Work with calculation agents
- [Calculation Endpoints](./calculations.md) - Calculate emissions
- [Quick Start Guide](../guides/quickstart.md) - Get started in 5 minutes
