# GreenLang API Documentation Overview

**Last Updated:** 2025-11-21
**Version:** 1.0.0

## Overview

GreenLang provides comprehensive REST and GraphQL APIs for climate compliance automation. This document provides an overview of all available APIs and their documentation.

## Available APIs

### 1. GreenLang Core API (REST)

**Production-grade emission factor queries and GHG calculations**

- **Base URL**: `https://api.greenlang.io`
- **Documentation**: [`greenlang/docs/api/openapi.yaml`](../greenlang/docs/api/openapi.yaml)
- **Interactive Docs**: `https://api.greenlang.io/api/docs`

**Key Features:**
- 327+ emission factors (US, EU, UK, global)
- Multi-gas breakdown (CO2, CH4, N2O)
- Scope 1/2/3 calculations
- Batch processing (up to 100 calculations)
- Full provenance tracking
- <50ms response time (95th percentile)

**Endpoints:**
- `GET /api/v1/factors` - List emission factors
- `GET /api/v1/factors/{factor_id}` - Get factor by ID
- `GET /api/v1/factors/search` - Search factors
- `POST /api/v1/calculate` - Calculate emissions
- `POST /api/v1/calculate/batch` - Batch calculations
- `GET /api/v1/health` - Health check
- `GET /api/v1/stats` - API statistics

**Rate Limits:**
- Factor Queries: 1000/minute
- Calculations: 500/minute
- Batch Calculations: 100/minute

---

### 2. GL-CBAM-APP API (REST)

**EU CBAM Transitional Registry Reporting**

- **Base URL**: `https://cbam-api.greenlang.io`
- **Documentation**: [`GL-CBAM-APP/CBAM-Importer-Copilot/docs/api/openapi.yaml`](../GL-CBAM-APP/CBAM-Importer-Copilot/docs/api/openapi.yaml)
- **Interactive Docs**: `https://cbam-api.greenlang.io/docs`

**Key Features:**
- Shipment intake (CSV, Excel, JSON, XML)
- Zero-hallucination emissions calculation
- CBAM registry report generation
- Full provenance for regulatory compliance
- Production monitoring (Prometheus metrics)

**Endpoints:**
- `GET /health` - Basic health check
- `GET /health/ready` - Readiness probe
- `GET /health/live` - Liveness probe
- `GET /metrics` - Prometheus metrics
- `POST /api/v1/pipeline/execute` - Execute CBAM pipeline
- `GET /api/v1/info` - Application info

**Rate Limits:**
- Pipeline Execution: 10/minute
- General Endpoints: 100/minute

**SLA:**
- Availability: 99.9%
- Success Rate: 99%
- Latency p95: <10 minutes

---

### 3. GL-CSRD-APP API (REST)

**CSRD/ESRS Digital Reporting Platform**

- **Base URL**: `https://csrd-api.greenlang.io`
- **Documentation**: [`GL-CSRD-APP/CSRD-Reporting-Platform/docs/api/openapi.yaml`](../GL-CSRD-APP/CSRD-Reporting-Platform/docs/api/openapi.yaml)
- **Interactive Docs**: `https://csrd-api.greenlang.io/docs`

**Key Features:**
- 975+ ESRS metrics calculation
- Double materiality assessment (AI-powered)
- XBRL/iXBRL report generation
- Multi-framework integration (CSRD, GRI, SASB, TCFD)
- Zero-hallucination calculations

**6-Agent Pipeline:**
1. IntakeAgent - Data validation
2. CalculatorAgent - 975 metric calculations
3. MaterialityAgent - Double materiality
4. AuditAgent - Compliance checks
5. ReportingAgent - XBRL generation
6. AggregatorAgent - Multi-framework

**Endpoints:**
- `POST /api/v1/pipeline/run` - Execute pipeline
- `GET /api/v1/pipeline/status/{job_id}` - Job status
- `GET /api/v1/pipeline/jobs` - List jobs
- `POST /api/v1/validate` - Validate data
- `POST /api/v1/calculate/{metric_id}` - Calculate metric
- `POST /api/v1/report/generate` - Generate report
- `POST /api/v1/materiality/assess` - Materiality assessment
- `GET /health` - Health check
- `GET /ready` - Readiness check
- `GET /metrics` - Prometheus metrics

**Rate Limits:**
- Pipeline Execution: 10/minute
- Validation: 60/minute
- General Queries: 100/minute

**SLA:**
- Availability: 99.9%
- Success Rate: 99%
- Latency p95: <15 minutes

---

### 4. GL-VCCI-Carbon-APP API (REST)

**Scope 3 Carbon Intelligence Platform**

- **Base URL**: `https://vcci-api.greenlang.io`
- **Documentation**: [`GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/docs/api/openapi.yaml`](../GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/docs/api/openapi.yaml)
- **Interactive Docs**: `https://vcci-api.greenlang.io/docs`

**Key Features:**
- All 15 Scope 3 categories
- AI-powered hotspot analysis
- Automated supplier engagement
- ERP integration (SAP, Oracle, NetSuite)
- Circuit breaker for resilient external calls
- JWT authentication with multi-tenancy

**5-Agent Pipeline:**
1. IntakeAgent - Supplier data validation
2. CalculatorAgent - Multi-method emissions calculation
3. HotspotAgent - ML-powered hotspot detection
4. EngagementAgent - Automated supplier outreach
5. ReportingAgent - CDP/TCFD/SBTi reports

**Endpoints:**
- `POST /api/v1/intake/suppliers` - Submit supplier data
- `POST /api/v1/calculator/calculate` - Calculate Scope 3
- `POST /api/v1/hotspot/analyze` - Hotspot analysis
- `POST /api/v1/engagement/campaigns` - Create campaign
- `POST /api/v1/reporting/generate` - Generate report
- `GET /api/v1/factors/search` - Search factors
- `GET /api/v1/methodologies` - List methodologies
- `POST /api/v1/connectors/sap/sync` - SAP sync
- `GET /health/live` - Liveness probe
- `GET /health/ready` - Readiness probe
- `GET /health/detailed` - Detailed health (circuit breakers)
- `GET /metrics` - Prometheus metrics

**Rate Limits (per user):**
- Supplier Intake: 100/minute
- Calculations: 500/minute
- Hotspot Analysis: 100/minute
- Engagement: 50/minute
- Reporting: 20/minute
- Factor Queries: 1000/minute

**SLA:**
- Availability: 99.9%
- Success Rate: 99%
- Latency p95: <5 seconds
- Security: SOC 2 Type II

---

### 5. GreenLang GraphQL API

**Type-safe agent orchestration and workflow management**

- **Endpoint**: `https://api.greenlang.io/graphql`
- **Documentation**: [`greenlang/docs/api/graphql-schema.md`](../greenlang/docs/api/graphql-schema.md)
- **Playground**: `https://api.greenlang.io/graphql/playground`
- **Schema**: [`greenlang/api/graphql/schema.graphql`](../greenlang/api/graphql/schema.graphql)

**Key Features:**
- Strongly typed schema
- Real-time subscriptions (WebSocket)
- Relay-style pagination
- RBAC integration
- Query complexity limiting
- Batch operations

**Core Types:**
- **Agent**: Climate compliance agents
- **Workflow**: Multi-agent pipelines
- **Execution**: Agent/workflow runs
- **User**: User management
- **Role**: RBAC roles
- **Permission**: Access control

**Operations:**
- **Queries**: Get agents, workflows, executions, system health
- **Mutations**: Create/update/delete resources, execute workflows
- **Subscriptions**: Real-time execution updates, system metrics

**Rate Limits:**
- Queries: 1000/minute
- Mutations: 500/minute
- Subscriptions: 100 concurrent
- Query Complexity: Max 1000 points

---

## Authentication

All APIs use Bearer token authentication:

```
Authorization: Bearer <token>
```

### Token Types

1. **JWT Tokens**: For user authentication
   - Obtain from auth service
   - Include user identity and roles
   - Expire after configured TTL

2. **API Keys**: For service-to-service
   - Generated in admin portal
   - Never expire (revoke manually)
   - Scoped to specific permissions

### Example: cURL

```bash
curl -X GET "https://api.greenlang.io/api/v1/factors" \
  -H "Authorization: Bearer <your-token>" \
  -H "Content-Type: application/json"
```

### Example: JavaScript

```javascript
const response = await fetch('https://api.greenlang.io/api/v1/factors', {
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  }
});
```

### Example: Python

```python
import requests

headers = {
    'Authorization': f'Bearer {token}',
    'Content-Type': 'application/json'
}

response = requests.get(
    'https://api.greenlang.io/api/v1/factors',
    headers=headers
)
```

---

## Health Checks

All APIs provide Kubernetes-compatible health endpoints:

### Liveness Probe
Indicates if the application is running (restart if false)

```
GET /health/live
GET /health
```

### Readiness Probe
Indicates if the application can serve traffic

```
GET /health/ready
GET /ready
```

### Startup Probe
Indicates if the application has completed initialization

```
GET /health/startup
```

### Detailed Health
Comprehensive health with dependency status

```
GET /health/detailed
```

**Response includes:**
- Overall status (healthy, degraded, unhealthy)
- Database connectivity and latency
- Redis connectivity and latency
- Circuit breaker states
- External dependency status

---

## Monitoring & Observability

### Prometheus Metrics

All APIs expose Prometheus metrics:

```
GET /metrics
```

**Available Metrics:**
- Request counters (by endpoint, status, method)
- Request duration histograms
- Active requests gauge
- Pipeline execution metrics
- Carbon-specific metrics (emissions, suppliers, calculations)
- System metrics (CPU, memory, connections)
- Circuit breaker states

### Structured Logging

All APIs use structured JSON logging with:
- Correlation IDs for request tracking
- Carbon context (emissions, suppliers, categories)
- User context (user_id, tenant_id)
- Performance metrics

### Distributed Tracing

Correlation IDs propagate across services:

```
X-Correlation-ID: 550e8400-e29b-41d4-a716-446655440000
```

---

## Rate Limiting

### Per-Endpoint Limits

| API | Endpoint Type | Rate Limit |
|-----|--------------|-----------|
| Core API | Factor Queries | 1000/min |
| Core API | Calculations | 500/min |
| Core API | Batch Calculations | 100/min |
| CBAM API | Pipeline Execution | 10/min |
| CSRD API | Pipeline Execution | 10/min |
| CSRD API | Validation | 60/min |
| VCCI API | Calculations | 500/min (per user) |
| VCCI API | Hotspot Analysis | 100/min (per user) |
| VCCI API | Reporting | 20/min (per user) |
| GraphQL | Queries | 1000/min |
| GraphQL | Mutations | 500/min |

### Rate Limit Headers

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1637510400
```

### Rate Limit Exceeded

```
HTTP/1.1 429 Too Many Requests
Retry-After: 60

{
  "error": "rate_limit_exceeded",
  "message": "Rate limit of 1000 requests per minute exceeded",
  "retry_after_seconds": 60
}
```

---

## Error Handling

### Standard Error Response

```json
{
  "error": "error_code",
  "message": "Human-readable error message",
  "details": {
    "field": "value",
    "additional": "context"
  },
  "timestamp": "2025-11-21T10:30:00Z",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request succeeded |
| 201 | Created | Resource created |
| 202 | Accepted | Request accepted (async) |
| 400 | Bad Request | Invalid request data |
| 401 | Unauthorized | Missing or invalid token |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service unhealthy |

---

## OpenAPI Specifications

All REST APIs provide OpenAPI 3.0 specifications:

- **GreenLang Core**: [`greenlang/docs/api/openapi.yaml`](../greenlang/docs/api/openapi.yaml)
- **GL-CBAM-APP**: [`GL-CBAM-APP/CBAM-Importer-Copilot/docs/api/openapi.yaml`](../GL-CBAM-APP/CBAM-Importer-Copilot/docs/api/openapi.yaml)
- **GL-CSRD-APP**: [`GL-CSRD-APP/CSRD-Reporting-Platform/docs/api/openapi.yaml`](../GL-CSRD-APP/CSRD-Reporting-Platform/docs/api/openapi.yaml)
- **GL-VCCI-Carbon**: [`GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/docs/api/openapi.yaml`](../GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/docs/api/openapi.yaml)

### Using OpenAPI Specs

**Generate Client SDKs:**

```bash
# Python
openapi-generator generate \
  -i openapi.yaml \
  -g python \
  -o ./python-client

# TypeScript
openapi-generator generate \
  -i openapi.yaml \
  -g typescript-axios \
  -o ./typescript-client
```

**Import to Postman:**

1. File > Import
2. Select `openapi.yaml`
3. Postman will generate full collection

**Import to Swagger UI:**

```bash
docker run -p 8080:8080 \
  -e SWAGGER_JSON=/openapi.yaml \
  -v $(pwd)/openapi.yaml:/openapi.yaml \
  swaggerapi/swagger-ui
```

---

## SDKs & Client Libraries

### Official SDKs

- **Python**: `pip install greenlang-sdk`
- **JavaScript/TypeScript**: `npm install @greenlang/sdk`
- **Go**: `go get github.com/greenlang/greenlang-go`

### Example: Python SDK

```python
from greenlang import GreenLangClient

client = GreenLangClient(
    api_key="your-api-key",
    base_url="https://api.greenlang.io"
)

# Calculate emissions
result = client.calculate_emissions(
    fuel_type="diesel",
    amount=100,
    unit="gallons",
    geography="US"
)

print(f"Total CO2e: {result.emissions_kg_co2e} kg")
```

### Example: TypeScript SDK

```typescript
import { GreenLangClient } from '@greenlang/sdk';

const client = new GreenLangClient({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.greenlang.io'
});

// Calculate emissions
const result = await client.calculateEmissions({
  fuelType: 'diesel',
  amount: 100,
  unit: 'gallons',
  geography: 'US'
});

console.log(`Total CO2e: ${result.emissionsKgCo2e} kg`);
```

---

## Support & Resources

### Documentation

- **API Overview**: This document
- **OpenAPI Specs**: See links above
- **GraphQL Schema**: [`greenlang/docs/api/graphql-schema.md`](../greenlang/docs/api/graphql-schema.md)
- **Interactive Docs**: Available at each API's `/docs` endpoint

### Interactive Explorers

- **Swagger UI**: `https://<api-host>/docs`
- **ReDoc**: `https://<api-host>/redoc`
- **GraphQL Playground**: `https://api.greenlang.io/graphql/playground`

### Support Channels

- **Email**: support@greenlang.io
- **Documentation**: https://docs.greenlang.io
- **GitHub Issues**: https://github.com/greenlang/greenlang/issues
- **Slack Community**: https://greenlang.slack.com

### Status Page

Monitor API uptime and incidents:

https://status.greenlang.io

---

## Changelog

### 2025-11-21 - v1.0.0

- Initial OpenAPI spec generation for all APIs
- GraphQL schema documentation
- Comprehensive API overview document

---

**Questions?** Contact support@greenlang.io
