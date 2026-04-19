# Agent Registry API Specification

**Version:** 1.0.0
**Status:** PRODUCTION
**Owner:** GL-DevOpsEngineer
**Last Updated:** 2025-12-03

---

## Overview

The Agent Registry API provides programmatic access to agent lifecycle management, versioning, discovery, and governance. This specification defines RESTful HTTP and gRPC endpoints for all registry operations.

**Base URL (REST):** `https://registry.greenlang.ai/api/v1`
**gRPC Endpoint:** `registry.greenlang.ai:443`

---

## Authentication

All API requests require authentication:

```http
Authorization: Bearer <api_key>
```

API keys are scoped to specific tenants and operations:
- `registry:read` - Read agent metadata
- `registry:write` - Publish new agents
- `registry:promote` - Promote agents between lifecycle states
- `registry:admin` - Full administrative access

---

## API: Publish Agent

Publish a new agent version to the registry.

### REST API

```http
POST /api/v1/registry/agents
Content-Type: application/json
Authorization: Bearer <api_key>

{
  "agent_id": "gl-cbam-calculator-v2",
  "name": "CBAM Carbon Calculator",
  "version": "2.3.1",
  "description": "Calculates embedded carbon for CBAM shipments",
  "domain": "sustainability.cbam",
  "type": "calculator",
  "tags": ["cbam", "carbon", "eu-regulation"],

  "container_image": {
    "registry": "gcr.io/greenlang",
    "image": "gcr.io/greenlang/cbam-calculator:2.3.1",
    "digest": "sha256:abc123...",
    "size_mb": 450
  },

  "runtime_requirements": {
    "cpu_request": "500m",
    "cpu_limit": "2000m",
    "memory_request": "512Mi",
    "memory_limit": "2Gi",
    "dependencies": {
      "services": [
        {
          "name": "greenlang.db",
          "version": ">=14.0",
          "required": true
        },
        {
          "name": "factor_broker",
          "version": ">=2.1",
          "required": true
        }
      ],
      "llm_providers": [
        {
          "provider": "anthropic",
          "models": ["claude-sonnet-4"],
          "required": true
        }
      ]
    }
  },

  "capabilities": [
    {
      "name": "calculate_carbon_intensity",
      "description": "Calculate embedded carbon per unit",
      "input_schema": "s3://greenlang-registry/schemas/cbam_input_v2.json",
      "output_schema": "s3://greenlang-registry/schemas/cbam_output_v2.json"
    }
  ],

  "artifacts": {
    "dockerfile": "s3://greenlang-registry/agents/cbam-calculator/2.3.1/Dockerfile",
    "readme": "s3://greenlang-registry/agents/cbam-calculator/2.3.1/README.md",
    "schemas": "s3://greenlang-registry/agents/cbam-calculator/2.3.1/schemas/"
  },

  "metadata": {
    "source_repo": "https://github.com/greenlang/agents/cbam-calculator",
    "commit_sha": "a1b2c3d4e5f6g7h8i9j0",
    "build_number": "1234"
  }
}
```

### Response

```json
{
  "status": "success",
  "version_id": "gl-cbam-calculator-v2@2.3.1",
  "lifecycle_state": "draft",
  "published_at": "2025-12-03T10:30:00Z",
  "registry_url": "https://registry.greenlang.ai/agents/gl-cbam-calculator-v2/2.3.1",
  "next_steps": {
    "evaluate": "gl agent evaluate gl-cbam-calculator-v2@2.3.1",
    "promote": "Evaluation required before promotion to experimental"
  }
}
```

### CLI Command

```bash
# Publish from current directory
gl agent publish \
  --name "CBAM Carbon Calculator" \
  --version 2.3.1 \
  --domain sustainability.cbam \
  --type calculator

# Publish from GitHub
gl agent publish \
  --from-git https://github.com/greenlang/agents/cbam-calculator \
  --tag v2.3.1
```

### gRPC

```protobuf
message PublishAgentRequest {
  string agent_id = 1;
  string name = 2;
  string version = 3;
  string description = 4;
  string domain = 5;
  string type = 6;
  repeated string tags = 7;
  ContainerImage container_image = 8;
  RuntimeRequirements runtime_requirements = 9;
  repeated Capability capabilities = 10;
  map<string, string> artifacts = 11;
  map<string, string> metadata = 12;
}

message PublishAgentResponse {
  string status = 1;
  string version_id = 2;
  string lifecycle_state = 3;
  string published_at = 4;
  string registry_url = 5;
}
```

---

## API: List Agents

List all agents visible to the current tenant.

### REST API

```http
GET /api/v1/registry/agents?domain=sustainability&state=certified&limit=50&offset=0
Authorization: Bearer <api_key>
```

### Query Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `domain` | string | Filter by domain | All |
| `type` | string | Filter by type | All |
| `state` | string | Filter by lifecycle state | All |
| `tags` | string | Comma-separated tags | All |
| `tenant_id` | string | Filter by tenant (admin only) | Current tenant |
| `limit` | integer | Max results (1-100) | 20 |
| `offset` | integer | Pagination offset | 0 |
| `sort_by` | string | Sort field (name, created_at, requests_30d) | created_at |
| `sort_order` | string | asc or desc | desc |

### Response

```json
{
  "total": 150,
  "limit": 50,
  "offset": 0,
  "agents": [
    {
      "agent_id": "gl-cbam-calculator-v2",
      "name": "CBAM Carbon Calculator",
      "latest_version": "2.3.1",
      "domain": "sustainability.cbam",
      "type": "calculator",
      "lifecycle_state": "certified",
      "created_at": "2025-11-01T08:00:00Z",
      "updated_at": "2025-11-15T12:00:00Z",
      "total_deployments": 42,
      "requests_30d": 15000000,
      "tags": ["cbam", "carbon", "eu-regulation"]
    },
    {
      "agent_id": "gl-csrd-materiality",
      "name": "CSRD Materiality Assessor",
      "latest_version": "1.5.2",
      "domain": "sustainability.csrd",
      "type": "assessor",
      "lifecycle_state": "certified",
      "created_at": "2025-10-15T10:00:00Z",
      "updated_at": "2025-11-20T14:00:00Z",
      "total_deployments": 28,
      "requests_30d": 8000000,
      "tags": ["csrd", "materiality", "eu-regulation"]
    }
  ]
}
```

### CLI Command

```bash
# List all certified agents
gl agent list --state certified

# List agents by domain
gl agent list --domain sustainability.cbam

# Search with tag filter
gl agent list --tags cbam,carbon
```

---

## API: Search Agents

Semantic search across agent capabilities.

### REST API

```http
POST /api/v1/registry/agents/search
Content-Type: application/json
Authorization: Bearer <api_key>

{
  "query": "Calculate embedded carbon for aluminum shipments",
  "filters": {
    "domain": "sustainability",
    "state": ["experimental", "certified"],
    "tags": ["carbon"]
  },
  "limit": 10,
  "similarity_threshold": 0.80
}
```

### Response

```json
{
  "query": "Calculate embedded carbon for aluminum shipments",
  "results": [
    {
      "agent_id": "gl-cbam-calculator-v2",
      "version": "2.3.1",
      "name": "CBAM Carbon Calculator",
      "similarity_score": 0.92,
      "matched_capabilities": [
        {
          "name": "calculate_carbon_intensity",
          "description": "Calculate embedded carbon per unit",
          "relevance": 0.95
        }
      ],
      "lifecycle_state": "certified"
    },
    {
      "agent_id": "gl-carbon-intensity-v1",
      "version": "1.2.0",
      "name": "Generic Carbon Intensity Calculator",
      "similarity_score": 0.87,
      "matched_capabilities": [
        {
          "name": "compute_emissions",
          "description": "Compute emissions for materials",
          "relevance": 0.88
        }
      ],
      "lifecycle_state": "experimental"
    }
  ]
}
```

### CLI Command

```bash
# Semantic search
gl agent search "Calculate embedded carbon for aluminum shipments"

# With filters
gl agent search "CSRD materiality assessment" --state certified --domain sustainability.csrd
```

---

## API: Get Agent Details

Retrieve full metadata for a specific agent.

### REST API

```http
GET /api/v1/registry/agents/{agent_id}
Authorization: Bearer <api_key>
```

### Response

```json
{
  "agent_id": "gl-cbam-calculator-v2",
  "name": "CBAM Carbon Calculator",
  "description": "Calculates embedded carbon for CBAM shipments",
  "domain": "sustainability.cbam",
  "type": "calculator",
  "tags": ["cbam", "carbon", "eu-regulation"],
  "created_by": "user@greenlang.ai",
  "team": "greenlang/cbam-team",
  "tenant_id": "customer-abc-123",
  "created_at": "2025-11-01T08:00:00Z",
  "updated_at": "2025-11-15T12:00:00Z",

  "latest_version": "2.3.1",
  "versions": [
    {
      "version": "2.3.1",
      "lifecycle_state": "certified",
      "published_at": "2025-11-15T12:00:00Z"
    },
    {
      "version": "2.3.0",
      "lifecycle_state": "certified",
      "published_at": "2025-11-10T10:00:00Z"
    },
    {
      "version": "2.2.5",
      "lifecycle_state": "deprecated",
      "published_at": "2025-10-20T14:00:00Z",
      "deprecated_at": "2025-11-15T12:00:00Z"
    }
  ],

  "usage_analytics": {
    "total_deployments": 42,
    "active_deployments": 38,
    "requests_30d": 15000000,
    "unique_tenants_30d": 25,
    "error_rate_30d": 0.004
  }
}
```

---

## API: Get Specific Version

Retrieve full metadata for a specific agent version.

### REST API

```http
GET /api/v1/registry/agents/{agent_id}/versions/{version}
Authorization: Bearer <api_key>
```

### Response

```json
{
  "version_id": "gl-cbam-calculator-v2@2.3.1",
  "agent_id": "gl-cbam-calculator-v2",
  "version": "2.3.1",
  "semantic_version": {
    "major": 2,
    "minor": 3,
    "patch": 1
  },

  "lifecycle_state": "certified",
  "state_history": [
    {
      "state": "draft",
      "entered_at": "2025-11-01T08:00:00Z",
      "duration_hours": 120
    },
    {
      "state": "experimental",
      "entered_at": "2025-11-06T08:00:00Z",
      "duration_hours": 72,
      "experimental_users": 5,
      "experimental_requests": 50000
    },
    {
      "state": "certified",
      "entered_at": "2025-11-15T12:00:00Z",
      "certified_by": "qa-team@greenlang.ai"
    }
  ],

  "container_image": {
    "registry": "gcr.io/greenlang",
    "image": "gcr.io/greenlang/cbam-calculator:2.3.1",
    "digest": "sha256:abc123...",
    "size_mb": 450
  },

  "runtime_requirements": {
    "cpu_request": "500m",
    "cpu_limit": "2000m",
    "memory_request": "512Mi",
    "memory_limit": "2Gi",
    "dependencies": {
      "services": [
        {
          "name": "greenlang.db",
          "version": ">=14.0",
          "required": true
        }
      ]
    }
  },

  "evaluation_results": {
    "evaluation_run_id": "eval-run-2025-11-15-001",
    "evaluated_at": "2025-11-15T12:00:00Z",
    "performance_metrics": {
      "latency_p50_ms": 120,
      "latency_p95_ms": 450,
      "latency_p99_ms": 850,
      "throughput_per_sec": 1200
    },
    "quality_metrics": {
      "accuracy": 0.98,
      "precision": 0.96,
      "recall": 0.94
    },
    "certification_status": {
      "certified": true,
      "certification_level": "production",
      "certification_date": "2025-11-15"
    }
  },

  "usage_metrics_30d": {
    "requests": 15000000,
    "error_rate": 0.004,
    "p95_latency_ms": 460,
    "active_deployments": 38
  }
}
```

### CLI Command

```bash
# Get latest version
gl agent get gl-cbam-calculator-v2

# Get specific version
gl agent get gl-cbam-calculator-v2@2.3.1

# Get version with evaluation results
gl agent get gl-cbam-calculator-v2@2.3.1 --include-evaluation
```

---

## API: Promote Agent

Promote an agent version to the next lifecycle state.

### REST API

```http
POST /api/v1/registry/agents/{agent_id}/versions/{version}/promote
Content-Type: application/json
Authorization: Bearer <api_key>

{
  "target_state": "certified",
  "reason": "All evaluation criteria met, production-ready",
  "metadata": {
    "certified_by": "qa-team@greenlang.ai",
    "evaluation_run_id": "eval-run-2025-11-15-001"
  }
}
```

### Promotion Paths

```
draft → experimental → certified
                      ↓
                  deprecated
```

### Promotion Criteria

**draft → experimental:**
- Container image published
- Basic evaluation passed
- No critical security vulnerabilities

**experimental → certified:**
- Comprehensive evaluation passed (performance, quality, compliance)
- Minimum experimental usage (>10K requests, >3 users)
- Error rate < 1%
- Security scan passed
- Documentation complete

### Response

```json
{
  "status": "success",
  "version_id": "gl-cbam-calculator-v2@2.3.1",
  "previous_state": "experimental",
  "current_state": "certified",
  "promoted_at": "2025-11-15T12:00:00Z",
  "promoted_by": "qa-team@greenlang.ai",
  "criteria_check": {
    "evaluation_passed": true,
    "security_scan_passed": true,
    "min_experimental_requests": true,
    "min_experimental_users": true,
    "error_rate_below_threshold": true,
    "documentation_complete": true
  }
}
```

### CLI Command

```bash
# Promote to experimental
gl agent promote gl-cbam-calculator-v2@2.3.1 --to experimental

# Promote to certified (with checks)
gl agent promote gl-cbam-calculator-v2@2.3.1 --to certified --reason "Production ready"

# Auto-promote if criteria met
gl agent promote gl-cbam-calculator-v2@2.3.1 --auto
```

---

## API: Deprecate Agent

Mark an agent version as deprecated.

### REST API

```http
POST /api/v1/registry/agents/{agent_id}/versions/{version}/deprecate
Content-Type: application/json
Authorization: Bearer <api_key>

{
  "reason": "Replaced by version 2.4.0 with improved performance",
  "replacement_version": "2.4.0",
  "sunset_date": "2026-03-01",
  "migration_guide": "https://docs.greenlang.ai/migration/2.3-to-2.4"
}
```

### Response

```json
{
  "status": "success",
  "version_id": "gl-cbam-calculator-v2@2.3.1",
  "previous_state": "certified",
  "current_state": "deprecated",
  "deprecated_at": "2025-12-03T10:00:00Z",
  "sunset_date": "2026-03-01",
  "replacement_version": "2.4.0",
  "active_deployments": 38,
  "notification_sent_to": [
    "customer-abc-123",
    "customer-xyz-456"
  ]
}
```

### CLI Command

```bash
# Deprecate version
gl agent deprecate gl-cbam-calculator-v2@2.3.1 \
  --replacement 2.4.0 \
  --sunset 2026-03-01 \
  --reason "Replaced by 2.4.0"

# Deprecate with grace period
gl agent deprecate gl-cbam-calculator-v2@2.3.1 \
  --sunset-in 90d
```

---

## API: List Versions

List all versions of an agent.

### REST API

```http
GET /api/v1/registry/agents/{agent_id}/versions?state=certified&limit=20
Authorization: Bearer <api_key>
```

### Response

```json
{
  "agent_id": "gl-cbam-calculator-v2",
  "total_versions": 15,
  "versions": [
    {
      "version": "2.3.1",
      "lifecycle_state": "certified",
      "published_at": "2025-11-15T12:00:00Z",
      "requests_30d": 15000000,
      "error_rate": 0.004
    },
    {
      "version": "2.3.0",
      "lifecycle_state": "certified",
      "published_at": "2025-11-10T10:00:00Z",
      "requests_30d": 5000000,
      "error_rate": 0.005
    },
    {
      "version": "2.2.5",
      "lifecycle_state": "deprecated",
      "published_at": "2025-10-20T14:00:00Z",
      "deprecated_at": "2025-11-15T12:00:00Z",
      "requests_30d": 500000,
      "replacement_version": "2.3.0"
    }
  ]
}
```

---

## API: Check Governance Policy

Check if an agent version is allowed for a specific tenant.

### REST API

```http
POST /api/v1/registry/governance/check
Content-Type: application/json
Authorization: Bearer <api_key>

{
  "tenant_id": "customer-abc-123",
  "agent_id": "gl-cbam-calculator-v2",
  "version": "2.3.1",
  "environment": "production",
  "operation": "deploy"
}
```

### Response

```json
{
  "allowed": true,
  "policy_id": "policy-prod-certified-only",
  "policy_name": "Production Certified Agents Only",
  "checks": [
    {
      "check": "lifecycle_state",
      "required": "certified",
      "actual": "certified",
      "passed": true
    },
    {
      "check": "tenant_whitelist",
      "required": "in whitelist",
      "actual": "customer-abc-123 in whitelist",
      "passed": true
    },
    {
      "check": "security_scan",
      "required": "passed",
      "actual": "passed",
      "passed": true
    }
  ]
}
```

### CLI Command

```bash
# Check policy compliance
gl agent check gl-cbam-calculator-v2@2.3.1 \
  --tenant customer-abc-123 \
  --environment production
```

---

## API: Get Usage Metrics

Get usage metrics for an agent version.

### REST API

```http
GET /api/v1/registry/agents/{agent_id}/versions/{version}/metrics?window=30d
Authorization: Bearer <api_key>
```

### Response

```json
{
  "version_id": "gl-cbam-calculator-v2@2.3.1",
  "window": "30d",
  "metrics": {
    "total_requests": 15000000,
    "unique_tenants": 25,
    "active_deployments": 38,
    "error_rate": 0.004,
    "performance": {
      "p50_latency_ms": 125,
      "p95_latency_ms": 460,
      "p99_latency_ms": 890
    },
    "top_errors": [
      {
        "error_type": "FactorNotFound",
        "count": 45000,
        "percentage": 0.003
      }
    ],
    "top_tenants": [
      {
        "tenant_id": "customer-xyz-456",
        "requests": 5000000,
        "error_rate": 0.002
      }
    ]
  }
}
```

---

## Error Responses

All API errors follow this format:

```json
{
  "error": {
    "code": "AGENT_NOT_FOUND",
    "message": "Agent 'gl-nonexistent' not found",
    "details": {
      "agent_id": "gl-nonexistent"
    },
    "request_id": "req-abc123",
    "timestamp": "2025-12-03T10:30:00Z"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `AGENT_NOT_FOUND` | 404 | Agent does not exist |
| `VERSION_NOT_FOUND` | 404 | Version does not exist |
| `INVALID_STATE_TRANSITION` | 400 | Invalid lifecycle state transition |
| `PROMOTION_CRITERIA_NOT_MET` | 400 | Promotion criteria not satisfied |
| `UNAUTHORIZED` | 401 | Invalid or missing API key |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `CONFLICT` | 409 | Version already exists |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Internal server error |

---

## Rate Limits

| Tier | Reads/min | Writes/min | Search/min |
|------|-----------|------------|------------|
| Standard | 1,000 | 100 | 100 |
| Enterprise | 10,000 | 1,000 | 1,000 |
| Admin | Unlimited | Unlimited | Unlimited |

Rate limit headers:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1638360000
```

---

## Pagination

All list endpoints support cursor-based pagination:

```http
GET /api/v1/registry/agents?limit=50&cursor=eyJhZ2VudF9pZCI6ImdsLWNiYW0tY2FsYyJ9
```

Response includes next cursor:
```json
{
  "agents": [...],
  "pagination": {
    "limit": 50,
    "has_more": true,
    "next_cursor": "eyJhZ2VudF9pZCI6ImdsLWNzcmQtbWF0ZXJpYWxpdHkifQ=="
  }
}
```

---

## Webhooks

Subscribe to registry events:

### Webhook Events

- `agent.published` - New agent version published
- `agent.promoted` - Agent promoted to new state
- `agent.deprecated` - Agent marked as deprecated
- `agent.evaluation.completed` - Evaluation finished

### Webhook Payload

```json
{
  "event": "agent.promoted",
  "timestamp": "2025-12-03T10:30:00Z",
  "data": {
    "version_id": "gl-cbam-calculator-v2@2.3.1",
    "agent_id": "gl-cbam-calculator-v2",
    "version": "2.3.1",
    "previous_state": "experimental",
    "current_state": "certified",
    "promoted_by": "qa-team@greenlang.ai"
  },
  "signature": "sha256=abc123..."
}
```

---

## SDK Examples

### Python

```python
from greenlang.registry import RegistryClient

client = RegistryClient(api_key="glang_live_xxx")

# Publish agent
version = client.publish_agent(
    agent_id="gl-cbam-calculator-v2",
    name="CBAM Carbon Calculator",
    version="2.3.1",
    container_image="gcr.io/greenlang/cbam-calculator:2.3.1",
    runtime_requirements={
        "cpu_request": "500m",
        "memory_request": "512Mi"
    }
)

# Search agents
results = client.search_agents(
    query="Calculate embedded carbon for aluminum",
    filters={"state": ["certified"]}
)

# Promote agent
client.promote_agent(
    agent_id="gl-cbam-calculator-v2",
    version="2.3.1",
    target_state="certified"
)
```

### TypeScript

```typescript
import { RegistryClient } from '@greenlang/registry-sdk';

const client = new RegistryClient({
  apiKey: 'glang_live_xxx'
});

// Publish agent
const version = await client.publishAgent({
  agentId: 'gl-cbam-calculator-v2',
  name: 'CBAM Carbon Calculator',
  version: '2.3.1',
  containerImage: 'gcr.io/greenlang/cbam-calculator:2.3.1'
});

// Search agents
const results = await client.searchAgents({
  query: 'Calculate embedded carbon for aluminum',
  filters: { state: ['certified'] }
});

// Promote agent
await client.promoteAgent({
  agentId: 'gl-cbam-calculator-v2',
  version: '2.3.1',
  targetState: 'certified'
});
```

---

## Related Documentation

- [Registry Overview](../architecture/00-REGISTRY_OVERVIEW.md)
- [Agent Lifecycle Management](../lifecycle/00-AGENT_LIFECYCLE.md)
- [Governance Controls](../governance/00-GOVERNANCE_CONTROLS.md)

---

**Questions or feedback?**
- Slack: #agent-registry-api
- Email: api-support@greenlang.ai
- API Status: https://status.greenlang.ai
