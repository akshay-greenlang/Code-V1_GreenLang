"""
OpenAPI Configuration and Documentation

Provides:
- OpenAPI 3.1 specification customization
- Redoc theme configuration
- Code samples for multiple languages
- Authentication and rate limiting documentation

Example:
    >>> from app.docs import configure_openapi
    >>> configure_openapi(app)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.requests import Request

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class OpenAPIConfig:
    """Configuration for OpenAPI documentation."""

    title: str = "GreenLang Agent Factory API"
    description: str = ""
    version: str = "1.0.0"
    terms_of_service: str = "https://greenlang.io/terms"
    contact_name: str = "GreenLang API Support"
    contact_email: str = "api-support@greenlang.io"
    contact_url: str = "https://greenlang.io/support"
    license_name: str = "Proprietary"
    license_url: str = "https://greenlang.io/license"
    servers: List[Dict[str, str]] = field(default_factory=lambda: [
        {"url": "https://api.greenlang.io", "description": "Production"},
        {"url": "https://api.staging.greenlang.io", "description": "Staging"},
        {"url": "http://localhost:8000", "description": "Development"},
    ])
    external_docs_url: str = "https://docs.greenlang.io"
    external_docs_description: str = "GreenLang Documentation"


# Default description
DEFAULT_DESCRIPTION = """
# GreenLang Agent Factory API

The Agent Factory API provides programmatic access to GreenLang's climate compliance platform.

## Overview

GreenLang enables enterprises to calculate, track, and report carbon emissions with
zero-hallucination deterministic calculations. Our Agent Factory provides:

- **Agent Registry**: Manage certified calculation agents
- **Execution Engine**: Run agents with full provenance tracking
- **Real-time Streaming**: WebSocket and GraphQL subscriptions
- **Batch Processing**: Bulk calculations via file upload
- **Webhooks**: Event-driven notifications

## Quick Start

### 1. Get API Key

Obtain an API key from the [GreenLang Dashboard](https://app.greenlang.io/settings/api-keys).

### 2. Make Your First Request

```bash
curl -X GET "https://api.greenlang.io/v1/agents" \\
  -H "X-API-Key: gl_your_api_key"
```

### 3. Execute an Agent

```python
import httpx

response = httpx.post(
    "https://api.greenlang.io/v1/agents/carbon/calculator/execute",
    headers={"X-API-Key": "gl_your_api_key"},
    json={
        "inputs": {
            "activity_type": "electricity",
            "quantity": 1000,
            "unit": "kWh",
            "region": "US"
        }
    }
)

result = response.json()
print(f"Carbon footprint: {result['outputs']['carbon_footprint']} tCO2e")
```

## Authentication

All API requests require authentication via one of:

| Method | Header | Format | Use Case |
|--------|--------|--------|----------|
| API Key | `X-API-Key` | `gl_xxxxx` | Server-to-server |
| JWT | `Authorization` | `Bearer eyJ...` | User sessions |

### API Key Authentication

```bash
curl -H "X-API-Key: gl_your_api_key" https://api.greenlang.io/v1/agents
```

### JWT Authentication

```bash
curl -H "Authorization: Bearer eyJhbG..." https://api.greenlang.io/v1/agents
```

## Rate Limiting

API requests are rate limited to ensure fair usage:

| Tier | Requests/Minute | Requests/Hour | Burst |
|------|-----------------|---------------|-------|
| Free | 60 | 1,000 | 10 |
| Pro | 300 | 10,000 | 50 |
| Enterprise | 1,000 | 100,000 | 200 |

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 300
X-RateLimit-Remaining: 299
X-RateLimit-Reset: 1699876543
```

When rate limited, you'll receive a `429 Too Many Requests` response with a
`Retry-After` header indicating when to retry.

## Errors

The API uses conventional HTTP status codes:

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid/missing authentication |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found |
| 429 | Too Many Requests - Rate limited |
| 500 | Internal Server Error |

Error responses follow a consistent format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input: quantity must be positive",
    "details": {
      "field": "quantity",
      "received": -100
    }
  }
}
```

## Versioning

The API is versioned via URL path (`/v1/`, `/v2/`). We maintain backward
compatibility within major versions.

| Version | Status | End of Life |
|---------|--------|-------------|
| v2 | Current | - |
| v1 | Supported | 2025-12-31 |

## SDKs

Official SDKs are available for:

- [Python](https://github.com/greenlang/greenlang-python)
- [JavaScript/TypeScript](https://github.com/greenlang/greenlang-js)
- [Go](https://github.com/greenlang/greenlang-go)

## Support

- **Documentation**: [docs.greenlang.io](https://docs.greenlang.io)
- **API Status**: [status.greenlang.io](https://status.greenlang.io)
- **Support**: [support@greenlang.io](mailto:support@greenlang.io)
"""


# =============================================================================
# Code Samples
# =============================================================================


CODE_SAMPLES = {
    "execute_agent": {
        "python": '''import httpx

client = httpx.Client(
    base_url="https://api.greenlang.io",
    headers={"X-API-Key": "gl_your_api_key"}
)

response = client.post(
    "/v1/agents/carbon/calculator/execute",
    json={
        "inputs": {
            "activity_type": "electricity",
            "quantity": 1000,
            "unit": "kWh",
            "region": "US-WECC"
        }
    }
)

execution = response.json()
print(f"Execution ID: {execution['execution_id']}")
print(f"Carbon: {execution['outputs']['carbon_footprint']} tCO2e")''',

        "javascript": '''const response = await fetch(
  "https://api.greenlang.io/v1/agents/carbon/calculator/execute",
  {
    method: "POST",
    headers: {
      "X-API-Key": "gl_your_api_key",
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      inputs: {
        activity_type: "electricity",
        quantity: 1000,
        unit: "kWh",
        region: "US-WECC"
      }
    })
  }
);

const execution = await response.json();
console.log(`Carbon: ${execution.outputs.carbon_footprint} tCO2e`);''',

        "curl": '''curl -X POST "https://api.greenlang.io/v1/agents/carbon/calculator/execute" \\
  -H "X-API-Key: gl_your_api_key" \\
  -H "Content-Type: application/json" \\
  -d '{
    "inputs": {
      "activity_type": "electricity",
      "quantity": 1000,
      "unit": "kWh",
      "region": "US-WECC"
    }
  }' '''
    },
    "list_agents": {
        "python": '''import httpx

client = httpx.Client(
    base_url="https://api.greenlang.io",
    headers={"X-API-Key": "gl_your_api_key"}
)

response = client.get(
    "/v1/agents",
    params={"category": "emissions", "state": "CERTIFIED"}
)

agents = response.json()
for agent in agents["data"]:
    print(f"{agent['agent_id']}: {agent['name']}")''',

        "javascript": '''const response = await fetch(
  "https://api.greenlang.io/v1/agents?category=emissions&state=CERTIFIED",
  {
    headers: { "X-API-Key": "gl_your_api_key" }
  }
);

const { data: agents } = await response.json();
agents.forEach(agent => console.log(`${agent.agent_id}: ${agent.name}`));''',

        "curl": '''curl "https://api.greenlang.io/v1/agents?category=emissions&state=CERTIFIED" \\
  -H "X-API-Key: gl_your_api_key"'''
    },
    "websocket": {
        "python": '''import asyncio
import websockets
import json

async def monitor_execution(execution_id: str):
    uri = f"wss://api.greenlang.io/ws/connect?token=gl_your_api_key"

    async with websockets.connect(uri) as ws:
        # Subscribe to execution updates
        await ws.send(json.dumps({
            "type": "subscribe",
            "data": {"room": f"execution:{execution_id}"}
        }))

        async for message in ws:
            data = json.loads(message)
            if data["type"] == "execution.progress":
                print(f"Progress: {data['data']['progress_percent']}%")
            elif data["type"] == "execution.completed":
                print(f"Complete: {data['data']['result']}")
                break

asyncio.run(monitor_execution("exec-123"))''',

        "javascript": '''const ws = new WebSocket(
  "wss://api.greenlang.io/ws/connect?token=gl_your_api_key"
);

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: "subscribe",
    data: { room: "execution:exec-123" }
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  if (message.type === "execution.progress") {
    console.log(`Progress: ${message.data.progress_percent}%`);
  } else if (message.type === "execution.completed") {
    console.log("Complete:", message.data.result);
    ws.close();
  }
};'''
    },
    "batch_upload": {
        "python": '''import httpx

with open("emissions_data.csv", "rb") as f:
    response = httpx.post(
        "https://api.greenlang.io/v1/batch/jobs",
        headers={"X-API-Key": "gl_your_api_key"},
        files={"file": ("emissions_data.csv", f, "text/csv")},
        data={
            "config": json.dumps({
                "agent_id": "carbon/calculator",
                "input_mapping": {
                    "activity_type": "Activity",
                    "quantity": "Amount",
                    "unit": "Unit"
                },
                "output_format": "csv"
            })
        }
    )

job = response.json()
print(f"Job ID: {job['job_id']}")
print(f"Status: {job['status']}")''',

        "curl": '''curl -X POST "https://api.greenlang.io/v1/batch/jobs" \\
  -H "X-API-Key: gl_your_api_key" \\
  -F "file=@emissions_data.csv" \\
  -F 'config={"agent_id":"carbon/calculator","input_mapping":{"activity_type":"Activity","quantity":"Amount","unit":"Unit"},"output_format":"csv"}' '''
    }
}


# =============================================================================
# Security Schemes
# =============================================================================


SECURITY_SCHEMES = {
    "ApiKeyAuth": {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "API key for server-to-server authentication. Obtain from [Dashboard](https://app.greenlang.io/settings/api-keys).",
    },
    "BearerAuth": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "JWT token for user session authentication.",
    },
    "OAuth2": {
        "type": "oauth2",
        "flows": {
            "authorizationCode": {
                "authorizationUrl": "https://auth.greenlang.io/authorize",
                "tokenUrl": "https://auth.greenlang.io/token",
                "scopes": {
                    "read:agents": "Read agent information",
                    "write:agents": "Create and modify agents",
                    "execute:agents": "Execute agents",
                    "read:executions": "Read execution history",
                    "admin": "Full administrative access",
                }
            }
        }
    }
}


# =============================================================================
# Tags and Organization
# =============================================================================


TAGS_METADATA = [
    {
        "name": "Agents",
        "description": "Manage calculation agents - certified components that perform deterministic climate calculations.",
        "externalDocs": {
            "description": "Agent documentation",
            "url": "https://docs.greenlang.io/agents"
        }
    },
    {
        "name": "Executions",
        "description": "Execute agents and retrieve results with full provenance tracking.",
        "externalDocs": {
            "description": "Execution documentation",
            "url": "https://docs.greenlang.io/executions"
        }
    },
    {
        "name": "Search",
        "description": "Search and discover agents by capability, category, or text.",
    },
    {
        "name": "Metrics",
        "description": "Access usage metrics, performance data, and analytics.",
    },
    {
        "name": "Batch Processing",
        "description": "Process large datasets via file upload with async job tracking.",
        "externalDocs": {
            "description": "Batch processing guide",
            "url": "https://docs.greenlang.io/batch"
        }
    },
    {
        "name": "WebSocket",
        "description": "Real-time streaming for execution progress and metrics.",
        "externalDocs": {
            "description": "WebSocket documentation",
            "url": "https://docs.greenlang.io/websocket"
        }
    },
    {
        "name": "Webhooks",
        "description": "Configure webhooks for event-driven notifications.",
        "externalDocs": {
            "description": "Webhook documentation",
            "url": "https://docs.greenlang.io/webhooks"
        }
    },
    {
        "name": "Tenants",
        "description": "Multi-tenant organization management.",
    },
    {
        "name": "Audit",
        "description": "Access audit logs for compliance and security.",
    },
    {
        "name": "Health",
        "description": "Health check and readiness endpoints.",
    },
    {
        "name": "Gateway",
        "description": "API gateway endpoints for version and rate limit information.",
    },
]


# =============================================================================
# OpenAPI Customization
# =============================================================================


def get_openapi_spec(
    app: FastAPI,
    config: Optional[OpenAPIConfig] = None,
) -> Dict[str, Any]:
    """
    Generate customized OpenAPI specification.

    Args:
        app: FastAPI application
        config: Optional configuration

    Returns:
        OpenAPI specification dictionary
    """
    config = config or OpenAPIConfig()

    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=config.title,
        version=config.version,
        description=config.description or DEFAULT_DESCRIPTION,
        routes=app.routes,
        tags=TAGS_METADATA,
    )

    # Add info extensions
    openapi_schema["info"]["termsOfService"] = config.terms_of_service
    openapi_schema["info"]["contact"] = {
        "name": config.contact_name,
        "email": config.contact_email,
        "url": config.contact_url,
    }
    openapi_schema["info"]["license"] = {
        "name": config.license_name,
        "url": config.license_url,
    }
    openapi_schema["info"]["x-logo"] = {
        "url": "https://greenlang.io/logo.png",
        "altText": "GreenLang",
    }

    # Add servers
    openapi_schema["servers"] = config.servers

    # Add external docs
    openapi_schema["externalDocs"] = {
        "description": config.external_docs_description,
        "url": config.external_docs_url,
    }

    # Add security schemes
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    openapi_schema["components"]["securitySchemes"] = SECURITY_SCHEMES

    # Add global security (all endpoints require auth)
    openapi_schema["security"] = [
        {"ApiKeyAuth": []},
        {"BearerAuth": []},
    ]

    # Add code samples to endpoints
    _add_code_samples(openapi_schema)

    # Add rate limiting documentation
    _add_rate_limit_docs(openapi_schema)

    app.openapi_schema = openapi_schema
    return openapi_schema


def _add_code_samples(schema: Dict[str, Any]) -> None:
    """Add x-code-samples to relevant endpoints."""
    paths = schema.get("paths", {})

    # Map endpoints to code samples
    endpoint_samples = {
        "/v1/agents": "list_agents",
        "/v1/agents/{agent_id}/execute": "execute_agent",
        "/v1/batch/jobs": "batch_upload",
        "/ws/connect": "websocket",
    }

    for path, operations in paths.items():
        sample_key = endpoint_samples.get(path)
        if sample_key and sample_key in CODE_SAMPLES:
            for method, operation in operations.items():
                if isinstance(operation, dict):
                    operation["x-codeSamples"] = [
                        {"lang": lang, "label": lang.capitalize(), "source": code}
                        for lang, code in CODE_SAMPLES[sample_key].items()
                    ]


def _add_rate_limit_docs(schema: Dict[str, Any]) -> None:
    """Add rate limiting documentation to responses."""
    rate_limit_headers = {
        "X-RateLimit-Limit": {
            "description": "Maximum requests per minute",
            "schema": {"type": "integer"},
        },
        "X-RateLimit-Remaining": {
            "description": "Remaining requests in current window",
            "schema": {"type": "integer"},
        },
        "X-RateLimit-Reset": {
            "description": "Unix timestamp when rate limit resets",
            "schema": {"type": "integer"},
        },
    }

    # Add to all 200/201 responses
    for path, operations in schema.get("paths", {}).items():
        for method, operation in operations.items():
            if not isinstance(operation, dict):
                continue

            responses = operation.get("responses", {})
            for status_code in ["200", "201", "202"]:
                if status_code in responses:
                    if "headers" not in responses[status_code]:
                        responses[status_code]["headers"] = {}
                    responses[status_code]["headers"].update(rate_limit_headers)


# =============================================================================
# Custom Documentation Routes
# =============================================================================


def configure_openapi(
    app: FastAPI,
    config: Optional[OpenAPIConfig] = None,
    enable_swagger: bool = True,
    enable_redoc: bool = True,
) -> None:
    """
    Configure OpenAPI documentation for the application.

    Args:
        app: FastAPI application
        config: Optional configuration
        enable_swagger: Enable Swagger UI
        enable_redoc: Enable ReDoc

    Example:
        >>> from app.docs import configure_openapi
        >>> configure_openapi(app, OpenAPIConfig(title="My API"))
    """
    config = config or OpenAPIConfig()

    # Override OpenAPI schema
    def custom_openapi() -> Dict[str, Any]:
        return get_openapi_spec(app, config)

    app.openapi = custom_openapi

    # Custom Swagger UI
    if enable_swagger:
        @app.get("/docs", include_in_schema=False)
        async def custom_swagger_ui_html() -> HTMLResponse:
            return get_swagger_ui_html(
                openapi_url=app.openapi_url,
                title=f"{config.title} - Swagger UI",
                oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
                swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
                swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
                swagger_favicon_url="https://greenlang.io/favicon.ico",
            )

    # Custom ReDoc
    if enable_redoc:
        @app.get("/redoc", include_in_schema=False)
        async def custom_redoc_html() -> HTMLResponse:
            return get_redoc_html(
                openapi_url=app.openapi_url,
                title=f"{config.title} - Documentation",
                redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@latest/bundles/redoc.standalone.js",
                redoc_favicon_url="https://greenlang.io/favicon.ico",
                with_google_fonts=True,
            )

    # OpenAPI JSON endpoint with caching
    @app.get("/openapi.json", include_in_schema=False)
    async def get_openapi_json() -> JSONResponse:
        return JSONResponse(
            content=app.openapi(),
            headers={"Cache-Control": "public, max-age=3600"},
        )

    # API changelog
    @app.get("/api/changelog", include_in_schema=False)
    async def get_api_changelog() -> Dict[str, Any]:
        return {
            "changelog": [
                {
                    "version": "1.0.0",
                    "date": "2024-01-15",
                    "changes": [
                        "Initial API release",
                        "Agent CRUD operations",
                        "Execution engine",
                        "WebSocket support",
                    ]
                },
                {
                    "version": "1.1.0",
                    "date": "2024-03-01",
                    "changes": [
                        "Added batch processing",
                        "Webhook support",
                        "GraphQL API",
                        "Rate limit headers",
                    ]
                },
                {
                    "version": "1.2.0",
                    "date": "2024-06-01",
                    "changes": [
                        "Enhanced metrics",
                        "Audit logging",
                        "Multi-tenant improvements",
                    ]
                },
            ]
        }

    logger.info("OpenAPI documentation configured")


# =============================================================================
# Authentication Guide
# =============================================================================


AUTHENTICATION_GUIDE = """
# Authentication Guide

## Overview

The GreenLang API supports multiple authentication methods to accommodate
different use cases.

## API Key Authentication

API keys are the recommended authentication method for server-to-server
integrations.

### Obtaining an API Key

1. Log in to the [GreenLang Dashboard](https://app.greenlang.io)
2. Navigate to **Settings > API Keys**
3. Click **Create New Key**
4. Give your key a descriptive name
5. Select the appropriate scopes
6. Copy and securely store your key

**Important**: API keys are shown only once. Store them securely.

### Using API Keys

Include your API key in the `X-API-Key` header:

```bash
curl -H "X-API-Key: gl_your_api_key" https://api.greenlang.io/v1/agents
```

### API Key Security Best Practices

- Never commit API keys to source control
- Use environment variables or secret management systems
- Rotate keys regularly
- Use the minimum required scopes
- Set expiration dates when possible

## JWT Authentication

JWT tokens are used for user session authentication, typically in web
applications.

### Obtaining a JWT Token

```bash
curl -X POST "https://auth.greenlang.io/token" \\
  -d "grant_type=password" \\
  -d "username=user@example.com" \\
  -d "password=your_password" \\
  -d "client_id=your_client_id"
```

### Using JWT Tokens

Include the token in the `Authorization` header:

```bash
curl -H "Authorization: Bearer eyJhbG..." https://api.greenlang.io/v1/agents
```

### Token Refresh

JWT tokens expire after 24 hours. Use the refresh token to obtain a new
access token:

```bash
curl -X POST "https://auth.greenlang.io/token" \\
  -d "grant_type=refresh_token" \\
  -d "refresh_token=your_refresh_token" \\
  -d "client_id=your_client_id"
```

## OAuth 2.0

For third-party integrations, we support OAuth 2.0 authorization code flow.

See the [OAuth 2.0 Guide](https://docs.greenlang.io/oauth) for details.

## Scopes

| Scope | Description |
|-------|-------------|
| `read:agents` | Read agent information |
| `write:agents` | Create and modify agents |
| `execute:agents` | Execute agents |
| `read:executions` | Read execution history |
| `write:webhooks` | Manage webhooks |
| `admin` | Full administrative access |

## Troubleshooting

### 401 Unauthorized

- Verify your API key or token is correct
- Check that the key hasn't expired
- Ensure the key has the required scopes

### 403 Forbidden

- Your key doesn't have permission for this operation
- Check your assigned scopes and tenant access
"""


# =============================================================================
# Rate Limiting Guide
# =============================================================================


RATE_LIMITING_GUIDE = """
# Rate Limiting Guide

## Overview

The GreenLang API implements rate limiting to ensure fair usage and maintain
service quality for all users.

## Rate Limits by Tier

| Tier | Requests/Minute | Requests/Hour | Burst | Concurrent |
|------|-----------------|---------------|-------|------------|
| Free | 60 | 1,000 | 10 | 5 |
| Pro | 300 | 10,000 | 50 | 20 |
| Enterprise | 1,000 | 100,000 | 200 | 100 |

## Rate Limit Headers

All responses include rate limit information:

```
X-RateLimit-Limit: 300
X-RateLimit-Remaining: 299
X-RateLimit-Reset: 1699876543
```

| Header | Description |
|--------|-------------|
| `X-RateLimit-Limit` | Maximum requests per minute |
| `X-RateLimit-Remaining` | Remaining requests in current window |
| `X-RateLimit-Reset` | Unix timestamp when limit resets |

## Handling Rate Limits

When rate limited, you'll receive a `429 Too Many Requests` response:

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again in 45 seconds."
  }
}
```

The response includes a `Retry-After` header indicating seconds to wait.

### Best Practices

1. **Implement Exponential Backoff**

```python
import time
import httpx

def make_request_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        response = httpx.get(url, headers={"X-API-Key": api_key})

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            time.sleep(retry_after)
            continue

        return response

    raise Exception("Max retries exceeded")
```

2. **Monitor Rate Limit Headers**

```python
response = httpx.get(url, headers={"X-API-Key": api_key})

remaining = int(response.headers["X-RateLimit-Remaining"])
if remaining < 10:
    # Slow down requests
    time.sleep(1)
```

3. **Use Batch Endpoints**

For bulk operations, use the batch processing API instead of making
individual requests.

4. **Cache Responses**

Cache responses where appropriate to reduce API calls.

## Per-Endpoint Limits

Some endpoints have specific rate limits:

| Endpoint | Limit |
|----------|-------|
| `/v1/agents/*/execute` | 100/min |
| `/v1/batch/jobs` | 10/min |
| `/ws/connect` | 10 connections |

## Requesting Higher Limits

Enterprise customers can request custom rate limits. Contact
[sales@greenlang.io](mailto:sales@greenlang.io) to discuss your needs.
"""
