# {Endpoint Name} API

> {Brief description of what this endpoint does}

## Endpoint

```
{METHOD} /v1/{path}
```

## Description

{Detailed description of the endpoint functionality}

## Authentication

This endpoint requires authentication via one of:

- **API Key**: `X-API-Key: gl_xxxxx`
- **Bearer Token**: `Authorization: Bearer <jwt_token>`

## Rate Limiting

| Tier | Limit |
|------|-------|
| Free | 60/min |
| Pro | 300/min |
| Enterprise | 1000/min |

## Request

### Headers

| Header | Required | Description |
|--------|----------|-------------|
| `X-API-Key` | Yes* | API key authentication |
| `Authorization` | Yes* | Bearer token authentication |
| `Content-Type` | Yes | Must be `application/json` |
| `X-Request-ID` | No | Optional request tracking ID |

*One of `X-API-Key` or `Authorization` is required.

### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `{param}` | `string` | Yes | {Description} |

### Query Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `{param}` | `string` | No | `{default}` | {Description} |

### Request Body

```json
{
  "field_1": "string",
  "field_2": 0,
  "nested": {
    "option": true
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `field_1` | `string` | Yes | {Description} |
| `field_2` | `integer` | No | {Description} |
| `nested` | `object` | No | {Description} |

## Response

### Success Response (200 OK)

```json
{
  "data": {
    "id": "abc123",
    "result": 42.5,
    "metadata": {}
  },
  "meta": {
    "request_id": "req_123",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `data` | `object` | Response data |
| `data.id` | `string` | Resource identifier |
| `data.result` | `number` | Calculation result |
| `meta` | `object` | Response metadata |

### Error Responses

#### 400 Bad Request

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input: field_1 is required",
    "details": {
      "field": "field_1",
      "constraint": "required"
    }
  }
}
```

#### 401 Unauthorized

```json
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid or missing authentication"
  }
}
```

#### 404 Not Found

```json
{
  "error": {
    "code": "NOT_FOUND",
    "message": "Resource not found: {id}"
  }
}
```

#### 429 Too Many Requests

```json
{
  "error": {
    "code": "RATE_LIMITED",
    "message": "Rate limit exceeded",
    "details": {
      "retry_after": 60
    }
  }
}
```

## Examples

### cURL

```bash
curl -X {METHOD} "https://api.greenlang.io/v1/{path}" \
  -H "X-API-Key: gl_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "field_1": "value",
    "field_2": 100
  }'
```

### Python

```python
import httpx

response = httpx.{method}(
    "https://api.greenlang.io/v1/{path}",
    headers={"X-API-Key": "gl_your_api_key"},
    json={
        "field_1": "value",
        "field_2": 100
    }
)

data = response.json()
print(data)
```

### JavaScript

```javascript
const response = await fetch('https://api.greenlang.io/v1/{path}', {
  method: '{METHOD}',
  headers: {
    'X-API-Key': 'gl_your_api_key',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    field_1: 'value',
    field_2: 100
  })
});

const data = await response.json();
console.log(data);
```

## Notes

- {Important note 1}
- {Important note 2}

## Related Endpoints

- [{Related Endpoint 1}](./related_1.md) - {Description}
- [{Related Endpoint 2}](./related_2.md) - {Description}

---

*API Version: 1.0*
*Last Updated: YYYY-MM-DD*
